import argparse


def _add_common_args(arg_parser):
    arg_parser.add_argument("--config", type=str)

    # Input
    arg_parser.add_argument("--types_path", type=str, help="Path to type specifications")

    # Preprocessing
    arg_parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer")
    arg_parser.add_argument("--max_span_size", type=int, default=6, help="Maximum span length")
    arg_parser.add_argument("--lowercase", action="store_true", default=False,
                            help="Lowercase input text during preprocessing")
    arg_parser.add_argument("--sampling_processes", type=int, default=4,
                            help="Number of sampling processes. Set to 0 to disable multiprocessing")

    # Model / Training / Evaluation
    arg_parser.add_argument("--model_path", type=str, help="Path to pretrained model or checkpoint directory")
    arg_parser.add_argument("--model_type", type=str, default="ader", help="Model type")
    arg_parser.add_argument("--cpu", action="store_true", default=False,
                            help="Use CPU even if CUDA is available")
    arg_parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation or prediction batch size")
    arg_parser.add_argument("--max_pairs", type=int, default=400,
                            help="Maximum entity pairs processed in each relation-classification chunk")
    arg_parser.add_argument("--rel_filter_threshold", type=float, default=0.6,
                            help="Threshold for retaining predicted relations")
    arg_parser.add_argument("--size_embedding", type=int, default=25,
                            help="Dimensionality of span-size embeddings")
    arg_parser.add_argument("--prop_drop", type=float, default=0.1,
                            help="Dropout probability used in ADER")
    arg_parser.add_argument("--freeze_transformer", action="store_true", default=False,
                            help="Freeze pretrained encoder parameters")
    arg_parser.add_argument("--no_overlapping", action="store_true", default=False,
                            help="Do not evaluate overlapping entities or relations with overlapping entities")

    # Misc
    arg_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    arg_parser.add_argument("--cache_path", type=str, default=None,
                            help="Cache path for Hugging Face transformer models")
    arg_parser.add_argument("--debug", action="store_true", default=False, help="Enable debugging mode")


def _add_logging_args(arg_parser):
    arg_parser.add_argument("--label", type=str, help="Run label used as the directory name for logs and models")
    arg_parser.add_argument("--log_path", type=str, help="Directory for training/evaluation logs")
    arg_parser.add_argument("--store_predictions", action="store_true", default=False,
                            help="Store predictions in the log directory")
    arg_parser.add_argument("--store_examples", action="store_true", default=False,
                            help="Store HTML evaluation examples in the log directory")
    arg_parser.add_argument("--example_count", type=int, default=None,
                            help="Number of evaluation examples to store")


def train_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument("--train_path", type=str, help="Path to training dataset")
    arg_parser.add_argument("--valid_path", type=str, help="Path to validation dataset")

    # Logging and checkpointing
    arg_parser.add_argument("--save_path", type=str, help="Directory for model checkpoints")
    arg_parser.add_argument("--init_eval", action="store_true", default=False,
                            help="Evaluate the validation set before training")
    arg_parser.add_argument("--save_optimizer", action="store_true", default=False,
                            help="Save optimizer state together with the model checkpoint")
    arg_parser.add_argument("--train_log_iter", type=int, default=100,
                            help="Log training progress every N iterations")
    arg_parser.add_argument("--final_eval", action="store_true", default=False,
                            help="Evaluate only after the final epoch instead of after every epoch")

    # Training
    arg_parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    arg_parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    arg_parser.add_argument("--neg_entity_count", type=int, default=200,
                            help="Number of negative entity samples per document")
    arg_parser.add_argument("--neg_relation_count", type=int, default=400,
                            help="Number of negative relation samples per document")
    arg_parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    arg_parser.add_argument("--lr_warmup", type=float, default=0.1,
                            help="Warm-up proportion of total training steps")
    arg_parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    arg_parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")

    _add_common_args(arg_parser)
    _add_logging_args(arg_parser)

    return arg_parser


def eval_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument("--dataset_path", type=str, help="Path to evaluation dataset")

    _add_common_args(arg_parser)
    _add_logging_args(arg_parser)

    return arg_parser


def predict_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument("--dataset_path", type=str, help="Path to prediction dataset")
    arg_parser.add_argument("--predictions_path", type=str, help="Path to store prediction results")
    arg_parser.add_argument("--spacy_model", type=str, help="SpaCy model used for tokenization")

    _add_common_args(arg_parser)

    return arg_parser
