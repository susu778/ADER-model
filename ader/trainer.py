"""
Base trainer utilities for ADER.

This module provides logging, checkpoint saving, TensorBoard support, device
selection, and reproducibility utilities shared by ADER training and evaluation.
"""

import argparse
import datetime
import logging
import os
import sys
from typing import Dict, List, Tuple

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizer

from ader import util
from ader.opt import tensorboardX

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class BaseTrainer:
    """Base class with common training, logging, and checkpoint utilities."""

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._debug = self._args.debug

        run_key = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if hasattr(args, "save_path"):
            self._save_path = os.path.join(self._args.save_path, self._args.label, run_key)
            util.create_directories_dir(self._save_path)

        if hasattr(args, "log_path"):
            self._log_path = os.path.join(self._args.log_path, self._args.label, run_key)
            util.create_directories_dir(self._log_path)

            self._log_paths = {}
            self._logger = logging.getLogger()
            util.reset_logger(self._logger)

            log_formatter = logging.Formatter(
                "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
            )

            file_handler = logging.FileHandler(os.path.join(self._log_path, "all.log"), encoding="utf-8")
            file_handler.setFormatter(log_formatter)
            self._logger.addHandler(file_handler)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(log_formatter)
            self._logger.addHandler(console_handler)

            self._logger.setLevel(logging.DEBUG if self._debug else logging.INFO)

            self._summary_writer = tensorboardX.SummaryWriter(self._log_path) if tensorboardX is not None else None
            self._log_arguments()

        self._best_results = {}

        self._device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self._gpu_count = torch.cuda.device_count()

        if args.seed is not None:
            util.set_seed(args.seed)

    def _add_dataset_logging(self, *labels, data: Dict[str, List[str]]):
        for label in labels:
            log_paths = {}

            for key, columns in data.items():
                path = os.path.join(self._log_path, "%s_%s.csv" % (key, label))
                util.create_csv(path, *columns)
                log_paths[key] = path

            self._log_paths[label] = log_paths
            self._best_results[label] = 0

    def _log_arguments(self):
        util.save_dict(self._log_path, self._args, "args")
        if self._summary_writer is not None:
            util.summarize_dict(self._summary_writer, self._args, "args")

    def _log_tensorboard(self, dataset_label: str, data_label: str, data: object, iteration: int):
        if self._summary_writer is not None:
            self._summary_writer.add_scalar("data/%s/%s" % (dataset_label, data_label), data, iteration)

    def _log_csv(self, dataset_label: str, data_label: str, *data: Tuple[object]):
        logs = self._log_paths[dataset_label]
        util.append_csv(logs[data_label], *data)

    def _save_best(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: Optimizer,
        accuracy: float,
        iteration: int,
        label: str,
        extra=None,
    ):
        if accuracy > self._best_results[label]:
            self._logger.info("[%s] Best model at iteration %s: %.4f", label, iteration, accuracy)
            self._save_model(
                self._save_path,
                model,
                tokenizer,
                iteration,
                optimizer=optimizer if getattr(self._args, "save_optimizer", False) else None,
                save_as_best=True,
                name="model_%s" % label,
                extra=extra,
            )
            self._best_results[label] = accuracy

    def _save_model(
        self,
        save_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        iteration: int,
        optimizer: Optimizer = None,
        save_as_best: bool = False,
        extra: dict = None,
        include_iteration: int = True,
        name: str = "model",
    ):
        extra_state = {"iteration": iteration}

        if optimizer:
            extra_state["optimizer"] = optimizer.state_dict()

        if extra:
            extra_state.update(extra)

        if save_as_best:
            dir_path = os.path.join(save_path, "%s_best" % name)
        else:
            dir_name = "%s_%s" % (name, iteration) if include_iteration else name
            dir_path = os.path.join(save_path, dir_name)

        util.create_directories_dir(dir_path)

        if isinstance(model, DataParallel):
            model.module.save_pretrained(dir_path)
        else:
            model.save_pretrained(dir_path)

        tokenizer.save_pretrained(dir_path)

        state_path = os.path.join(dir_path, "extra.state")
        torch.save(extra_state, state_path)

    def _get_lr(self, optimizer):
        return [group["lr"] for group in optimizer.param_groups]

    def _close_summary_writer(self):
        if self._summary_writer is not None:
            self._summary_writer.close()