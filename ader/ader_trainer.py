"""
Training, evaluation, and prediction utilities for ADER.

This trainer handles model loading, optimization, validation-based checkpoint
selection, strict evaluation, and prediction export for the ADER architecture.
"""

import argparse
import math
import os
from typing import Type

import torch
import transformers
from torch.nn import DataParallel
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from ader import models
from ader import prediction
from ader import sampling
from ader import util
from ader.entities import Dataset
from ader.evaluator import Evaluator
from ader.input_reader import BaseInputReader
from ader.loss import ADERLoss, Loss
from ader.trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class ADERTrainer(BaseTrainer):
    """Trainer for joint entity and directed relation extraction with ADER."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        cache_dir = getattr(args, "cache_path", None)
        self._tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_path,
            do_lower_case=args.lowercase,
            cache_dir=cache_dir,
        )

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        train_label, valid_label = "train", "valid"

        self._logger.info("Datasets: %s, %s", train_path, valid_path)
        self._logger.info("Model type: %s", args.model_type)

        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        input_reader = input_reader_cls(
            types_path,
            self._tokenizer,
            args.neg_entity_count,
            args.neg_relation_count,
            args.max_span_size,
            self._logger,
        )
        train_dataset = input_reader.read(train_path, train_label)
        validation_dataset = input_reader.read(valid_path, valid_label)
        self._log_datasets(input_reader)

        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        self._logger.info("Updates per epoch: %s", updates_epoch)
        self._logger.info("Updates total: %s", updates_total)

        model = self._load_model(input_reader)
        model.to(self._device)

        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.lr_warmup * updates_total),
            num_training_steps=updates_total,
        )

        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        entity_criterion = torch.nn.CrossEntropyLoss(reduction="none")
        compute_loss = ADERLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        for epoch in range(args.epochs):
            epoch_iteration = self._train_epoch(
                model=model,
                compute_loss=compute_loss,
                optimizer=optimizer,
                dataset=train_dataset,
                updates_epoch=updates_epoch,
                epoch=epoch,
            )

            if not args.final_eval or (epoch == args.epochs - 1):
                _, _, rel_nec_eval = self._eval(
                    model,
                    validation_dataset,
                    input_reader,
                    epoch + 1,
                    updates_epoch,
                )

                # Strict end-to-end relation F1 with entity types and relation direction.
                strict_relation_f1 = rel_nec_eval[2]
                global_iteration = (epoch + 1) * updates_epoch

                extra = {
                    "epoch": epoch + 1,
                    "updates_epoch": updates_epoch,
                    "epoch_iteration": epoch_iteration,
                    "valid_relation_f1_strict": strict_relation_f1,
                    "seed": args.seed,
                }

                self._save_best(
                    model=model,
                    tokenizer=self._tokenizer,
                    optimizer=optimizer,
                    accuracy=strict_relation_f1,
                    iteration=global_iteration,
                    label=valid_label,
                    extra=extra,
                )

        extra = {"epoch": args.epochs, "updates_epoch": updates_epoch, "epoch_iteration": 0}
        global_iteration = args.epochs * updates_epoch
        self._save_model(
            self._save_path,
            model,
            self._tokenizer,
            global_iteration,
            optimizer=optimizer if getattr(self._args, "save_optimizer", False) else None,
            extra=extra,
            include_iteration=False,
            name="final_model",
        )

        self._logger.info("Logged in: %s", self._log_path)
        self._logger.info("Saved in: %s", self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        dataset_label = "test"

        self._logger.info("Dataset: %s", dataset_path)
        self._logger.info("Model: %s", args.model_type)

        self._init_eval_logging(dataset_label)

        input_reader = input_reader_cls(
            types_path,
            self._tokenizer,
            max_span_size=args.max_span_size,
            logger=self._logger,
        )
        test_dataset = input_reader.read(dataset_path, dataset_label)
        self._log_datasets(input_reader)

        model = self._load_model(input_reader)
        model.to(self._device)

        self._eval(model, test_dataset, input_reader)

        self._logger.info("Logged in: %s", self._log_path)
        self._close_summary_writer()

    def predict(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args

        input_reader = input_reader_cls(
            types_path,
            self._tokenizer,
            max_span_size=args.max_span_size,
            spacy_model=args.spacy_model,
        )
        dataset = input_reader.read(dataset_path, "dataset")

        model = self._load_model(input_reader)
        model.to(self._device)

        self._predict(model, dataset, input_reader)

    def _load_model(self, input_reader: BaseInputReader):
        model_class = models.get_model(self._args.model_type)
        cache_dir = getattr(self._args, "cache_path", None)

        config = BertConfig.from_pretrained(self._args.model_path, cache_dir=cache_dir)

        # Store ADER-specific metadata for checkpoints and downstream loading.
        config.ader_version = model_class.VERSION
        config.spert_version = model_class.VERSION  # Legacy compatibility for SpERT-derived utilities.
        config.relation_types = input_reader.relation_type_count - 1
        config.entity_types = input_reader.entity_type_count
        config.size_embedding = self._args.size_embedding
        config.prop_drop = self._args.prop_drop
        config.freeze_transformer = self._args.freeze_transformer
        config.max_pairs = self._args.max_pairs

        model = model_class.from_pretrained(
            self._args.model_path,
            config=config,
            cls_token=self._tokenizer.cls_token_id,
            relation_types=config.relation_types,
            entity_types=config.entity_types,
            size_embedding=config.size_embedding,
            prop_drop=config.prop_drop,
            freeze_transformer=config.freeze_transformer,
            max_pairs=config.max_pairs,
            cache_dir=cache_dir,
        )

        return model

    def _train_epoch(
        self,
        model: torch.nn.Module,
        compute_loss: Loss,
        optimizer: Optimizer,
        dataset: Dataset,
        updates_epoch: int,
        epoch: int,
    ):
        self._logger.info("Train epoch: %s", epoch)

        dataset.switch_mode(Dataset.TRAIN_MODE)

        data_generator = None
        worker_init_fn = None

        if self._args.seed is not None:
            data_generator = torch.Generator()
            data_generator.manual_seed(int(self._args.seed) + int(epoch))
            worker_init_fn = util.seed_worker

        data_loader = DataLoader(
            dataset,
            batch_size=self._args.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
            worker_init_fn=worker_init_fn,
            generator=data_generator,
            persistent_workers=False,
        )

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self._args.train_batch_size

        for batch in tqdm(data_loader, total=total, desc="Train epoch %s" % epoch):
            model.train()
            batch = util.to_device(batch, self._device)

            entity_logits, rel_logits = model(
                encodings=batch["encodings"],
                context_masks=batch["context_masks"],
                entity_masks=batch["entity_masks"],
                entity_sizes=batch["entity_sizes"],
                relations=batch["rels"],
                rel_masks=batch["rel_masks"],
            )

            batch_loss = compute_loss.compute(
                entity_logits=entity_logits,
                rel_logits=rel_logits,
                rel_types=batch["rel_types"],
                entity_types=batch["entity_types"],
                entity_sample_masks=batch["entity_sample_masks"],
                rel_sample_masks=batch["rel_sample_masks"],
            )

            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self._args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        input_reader: BaseInputReader,
        epoch: int = 0,
        updates_epoch: int = 0,
        iteration: int = 0,
    ):
        self._logger.info("Evaluate: %s", dataset.label)

        if isinstance(model, DataParallel):
            model = model.module

        predictions_path = os.path.join(self._log_path, f"predictions_{dataset.label}_epoch_{epoch}.json")
        examples_path = os.path.join(self._log_path, f"examples_%s_{dataset.label}_epoch_{epoch}.html")
        evaluator = Evaluator(
            dataset,
            input_reader,
            self._tokenizer,
            self._args.rel_filter_threshold,
            self._args.no_overlapping,
            predictions_path,
            examples_path,
            self._args.example_count,
        )

        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self._args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self._args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
        )

        with torch.no_grad():
            model.eval()

            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc="Evaluate epoch %s" % epoch):
                batch = util.to_device(batch, self._device)

                entity_clf, rel_clf, rels = model(
                    encodings=batch["encodings"],
                    context_masks=batch["context_masks"],
                    entity_masks=batch["entity_masks"],
                    entity_sizes=batch["entity_sizes"],
                    entity_spans=batch["entity_spans"],
                    entity_sample_masks=batch["entity_sample_masks"],
                    inference=True,
                )

                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()

        self._log_eval(
            *ner_eval,
            *rel_eval,
            *rel_nec_eval,
            epoch,
            iteration,
            global_iteration,
            dataset.label,
        )

        if self._args.store_predictions and not self._args.no_overlapping:
            evaluator.store_predictions()

        if self._args.store_examples:
            evaluator.store_examples()

        return ner_eval, rel_eval, rel_nec_eval

    def _predict(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader):
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self._args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self._args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
        )

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc="Predict"):
                batch = util.to_device(batch, self._device)

                entity_clf, rel_clf, rels = model(
                    encodings=batch["encodings"],
                    context_masks=batch["context_masks"],
                    entity_masks=batch["entity_masks"],
                    entity_sizes=batch["entity_sizes"],
                    entity_spans=batch["entity_spans"],
                    entity_sample_masks=batch["entity_sample_masks"],
                    inference=True,
                )

                batch_pred_entities, batch_pred_relations = prediction.convert_predictions(
                    entity_clf,
                    rel_clf,
                    rels,
                    batch,
                    self._args.rel_filter_threshold,
                    input_reader,
                )

                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)

        prediction.store_predictions(dataset.documents, pred_entities, pred_relations, self._args.predictions_path)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        optimizer_params = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": self._args.weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int, iteration: int, global_iteration: int,
                   label: str):
        avg_loss = loss / self._args.train_batch_size
        lr = self._get_lr(optimizer)[0]

        self._log_tensorboard(label, "loss", loss, global_iteration)
        self._log_tensorboard(label, "loss_avg", avg_loss, global_iteration)
        self._log_tensorboard(label, "lr", lr, global_iteration)

        self._log_csv(label, "loss", loss, epoch, iteration, global_iteration)
        self._log_csv(label, "loss_avg", avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, "lr", lr, epoch, iteration, global_iteration)

    def _log_eval(
        self,
        ner_prec_micro: float,
        ner_rec_micro: float,
        ner_f1_micro: float,
        ner_prec_macro: float,
        ner_rec_macro: float,
        ner_f1_macro: float,
        rel_prec_micro: float,
        rel_rec_micro: float,
        rel_f1_micro: float,
        rel_prec_macro: float,
        rel_rec_macro: float,
        rel_f1_macro: float,
        rel_nec_prec_micro: float,
        rel_nec_rec_micro: float,
        rel_nec_f1_micro: float,
        rel_nec_prec_macro: float,
        rel_nec_rec_macro: float,
        rel_nec_f1_macro: float,
        epoch: int,
        iteration: int,
        global_iteration: int,
        label: str,
    ):
        self._log_tensorboard(label, "eval/ner_prec_micro", ner_prec_micro, global_iteration)
        self._log_tensorboard(label, "eval/ner_recall_micro", ner_rec_micro, global_iteration)
        self._log_tensorboard(label, "eval/ner_f1_micro", ner_f1_micro, global_iteration)
        self._log_tensorboard(label, "eval/ner_prec_macro", ner_prec_macro, global_iteration)
        self._log_tensorboard(label, "eval/ner_recall_macro", ner_rec_macro, global_iteration)
        self._log_tensorboard(label, "eval/ner_f1_macro", ner_f1_macro, global_iteration)

        self._log_tensorboard(label, "eval/rel_prec_micro", rel_prec_micro, global_iteration)
        self._log_tensorboard(label, "eval/rel_recall_micro", rel_rec_micro, global_iteration)
        self._log_tensorboard(label, "eval/rel_f1_micro", rel_f1_micro, global_iteration)
        self._log_tensorboard(label, "eval/rel_prec_macro", rel_prec_macro, global_iteration)
        self._log_tensorboard(label, "eval/rel_recall_macro", rel_rec_macro, global_iteration)
        self._log_tensorboard(label, "eval/rel_f1_macro", rel_f1_macro, global_iteration)

        self._log_tensorboard(label, "eval/rel_nec_prec_micro", rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, "eval/rel_nec_recall_micro", rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, "eval/rel_nec_f1_micro", rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, "eval/rel_nec_prec_macro", rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, "eval/rel_nec_recall_macro", rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, "eval/rel_nec_f1_macro", rel_nec_f1_macro, global_iteration)

        self._log_csv(
            label,
            "eval",
            ner_prec_micro,
            ner_rec_micro,
            ner_f1_micro,
            ner_prec_macro,
            ner_rec_macro,
            ner_f1_macro,
            rel_prec_micro,
            rel_rec_micro,
            rel_f1_micro,
            rel_prec_macro,
            rel_rec_macro,
            rel_f1_macro,
            rel_nec_prec_micro,
            rel_nec_rec_micro,
            rel_nec_f1_micro,
            rel_nec_prec_macro,
            rel_nec_rec_macro,
            rel_nec_f1_macro,
            epoch,
            iteration,
            global_iteration,
        )

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s", input_reader.relation_type_count)
        self._logger.info("Entity type count: %s", input_reader.entity_type_count)

        self._logger.info("Entities:")
        for entity_type in input_reader.entity_types.values():
            self._logger.info("%s=%s", entity_type.verbose_name, entity_type.index)

        self._logger.info("Relations:")
        for relation_type in input_reader.relation_types.values():
            self._logger.info("%s=%s", relation_type.verbose_name, relation_type.index)

        for key, dataset in input_reader.datasets.items():
            self._logger.info("Dataset: %s", key)
            self._logger.info("Document count: %s", dataset.document_count)
            self._logger.info("Relation count: %s", dataset.relation_count)
            self._logger.info("Entity count: %s", dataset.entity_count)

    def _init_train_logging(self, label):
        self._add_dataset_logging(
            label,
            data={
                "lr": ["lr", "epoch", "iteration", "global_iteration"],
                "loss": ["loss", "epoch", "iteration", "global_iteration"],
                "loss_avg": ["loss_avg", "epoch", "iteration", "global_iteration"],
            },
        )

    def _init_eval_logging(self, label):
        self._add_dataset_logging(
            label,
            data={
                "eval": [
                    "ner_prec_micro",
                    "ner_rec_micro",
                    "ner_f1_micro",
                    "ner_prec_macro",
                    "ner_rec_macro",
                    "ner_f1_macro",
                    "rel_prec_micro",
                    "rel_rec_micro",
                    "rel_f1_micro",
                    "rel_prec_macro",
                    "rel_rec_macro",
                    "rel_f1_macro",
                    "rel_nec_prec_micro",
                    "rel_nec_rec_micro",
                    "rel_nec_f1_micro",
                    "rel_nec_prec_macro",
                    "rel_nec_rec_macro",
                    "rel_nec_f1_macro",
                    "epoch",
                    "iteration",
                    "global_iteration",
                ]
            },
        )


# Backward-compatible alias for old scripts.
SpERTTrainer = ADERTrainer