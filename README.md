# ADER: Alzheimer’s Disease Entity and Relation Extraction

ADER is a joint entity and directed relation extraction framework for Alzheimer’s disease (AD) literature. It is designed to extract fine-grained biomedical entities and direction-sensitive relations from PubMed abstracts, supporting downstream biomedical knowledge graph construction and evidence tracing.

The model integrates a biomedical pretrained encoder, span-based entity classification, decoupled head-tail cross-attention, and biaffine relation classification. It is particularly designed for asymmetric biomedical relations, such as `Upregulate`, `Downregulate`, `Promote`, `Inhibit`, `Target`, and `Alleviate`.

## Main Features

- Span-based biomedical entity recognition for multi-token biomedical entities.
- BioLinkBERT-based contextual encoding for AD-related biomedical literature.
- Decoupled head-tail cross-attention for role-specific relation context modeling.
- Biaffine relation classifier for ordered and direction-sensitive relation prediction.
- Support for strict end-to-end entity and relation extraction evaluation.
- Prediction output in JSON format for downstream knowledge graph construction.
- Configuration templates for training, evaluation, and prediction.

## Model Overview

ADER follows a span-based joint extraction framework. Given an input sentence, the model first encodes the text using a biomedical pretrained language model. Candidate spans are then enumerated and classified as biomedical entity types or non-entities. For relation extraction, ordered entity pairs are constructed from predicted entities.

Unlike models that use a shared representation for both relation arguments, ADER projects the head and tail entities into separate query representations. These queries independently attend to the sentence context through decoupled head-tail cross-attention. The resulting role-specific contextual representations are then combined with entity features and passed to a biaffine classifier for directed relation prediction.

The main architectural components are:

1. **Contextual Encoder**  
   BioLinkBERT or another BERT-style pretrained encoder is used to obtain contextualized token representations.

2. **Span-based Entity Classification**  
   Candidate spans are represented using span-pooled token features, sentence-level context, and span-width embeddings.

3. **Decoupled Head-Tail Cross-Attention**  
   Head and tail entities use independent query projections to retrieve role-specific contextual information.

4. **Biaffine Relation Classification**  
   Ordered entity-pair representations are scored using a biaffine classifier to capture asymmetric biomedical relations.

## Repository Structure

```text
ADER/
├── ader/
│   ├── __init__.py
│   ├── ader_trainer.py
│   ├── trainer.py
│   ├── models.py
│   ├── entities.py
│   ├── evaluator.py
│   ├── input_reader.py
│   ├── loss.py
│   ├── opt.py
│   ├── prediction.py
│   ├── sampling.py
│   └── util.py
│
├── configs/
│   ├── train.conf
│   ├── eval.conf
│   └── predict.conf
│
├── args.py
├── config_reader.py
├── run_ader.py
├── README.md
├── requirements.txt
├── LICENSE
└── NOTICE.md
```

## Dataset Availability

The AD-specific annotated dataset used in our study is **not included** in this repository.

This repository provides:

- the ADER model implementation;
- training, evaluation, and prediction code;
- configuration templates;
- inline examples of the expected data format;
- documentation for preparing ADER-compatible datasets.

The AD-specific annotated dataset, pretrained encoder weights, trained checkpoints, prediction outputs, and Neo4j graph data are not included in this repository.

## Data Format

ADER uses a sentence-level JSON format. Each sample contains tokens, entity annotations, and relation annotations.

Example:

```json
[
  {
    "tokens": ["YKY", "may", "improve", "AD", "by", "targeting", "Nrf2", "to", "inhibit", "neuronal", "ferroptosis"],
    "entities": [
      {"type": "Drug", "start": 0, "end": 1},
      {"type": "AD", "start": 3, "end": 4},
      {"type": "Protein", "start": 6, "end": 7},
      {"type": "Biological_Process", "start": 9, "end": 11}
    ],
    "relations": [
      {"type": "Alleviate", "head": 0, "tail": 1},
      {"type": "Target", "head": 0, "tail": 2},
      {"type": "Inhibit", "head": 0, "tail": 3},
      {"type": "Via", "head": 1, "tail": 2},
      {"type": "Via", "head": 1, "tail": 3}
    ]
  }
]
```

Entity indices in relations refer to the order of entities in the `entities` list. Entity spans follow the `[start, end)` convention, where `start` is inclusive and `end` is exclusive.

## Entity and Relation Schema

The entity and relation schema should be provided by the user through the `types_path` field in the configuration file. The schema should follow the structure shown below.

Example structure:

```json
{
  "entities": {
    "AD": {"short": "AD", "verbose": "Alzheimer's disease"},
    "Disease": {"short": "DIS", "verbose": "Disease"},
    "Protein": {"short": "PRO", "verbose": "Protein"},
    "Drug": {"short": "DRUG", "verbose": "Drug"},
    "Biological_Process": {"short": "BP", "verbose": "Biological process"}
  },
  "relations": {
    "Associate": {"short": "ASSOC", "verbose": "Associate", "symmetric": false},
    "Upregulate": {"short": "UP", "verbose": "Upregulate", "symmetric": false},
    "Downregulate": {"short": "DOWN", "verbose": "Downregulate", "symmetric": false},
    "Inhibit": {"short": "INHIBIT", "verbose": "Inhibit", "symmetric": false},
    "Alleviate": {"short": "ALLEV", "verbose": "Alleviate", "symmetric": false}
  }
}
```

## Setup

### Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Recommended packages include:

```text
torch
transformers
numpy
scikit-learn
tqdm
jinja2
spacy
safetensors
tensorboardX
```

For prediction from raw text, install a spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Pretrained Models

Pretrained encoder weights are not included in this repository.

The `data/models/` directory is not included in this repository. Users should create it locally before running training, evaluation, or prediction.

Expected structure:

```text
data/models/
├── BioLinkBERT/
├── PubMedBERT/
└── bert-base-cased/
```

For example, when using BioLinkBERT, the configuration file should contain:

```ini
model_path = ./data/models/BioLinkBERT
tokenizer_path = ./data/models/BioLinkBERT
```

## Training

Use the following command to train ADER:

```bash
python run_ader.py train --config configs/train.conf
```

Main configuration fields:

```ini
label = ADER
model_type = ader
model_path = ./data/models/BioLinkBERT
tokenizer_path = ./data/models/BioLinkBERT

train_path = ./data/datasets/train-example.json
valid_path = ./data/datasets/dev-example.json
types_path = ./data/datasets/types-example.json

train_batch_size = 4
eval_batch_size = 1
epochs = 80
lr = 3e-5
rel_filter_threshold = 0.6
max_span_size = 6

log_path = ./data/log
save_path = ./data/save
```

## Evaluation

Use the following command to evaluate a trained model:

```bash
python run_ader.py eval --config configs/eval.conf
```

Example configuration:

```ini
label = ADER
model_type = ader
model_path = ./data/save/ADER/model_valid_best
tokenizer_path = ./data/models/BioLinkBERT

dataset_path = ./data/datasets/test-example.json
types_path = ./data/datasets/types-example.json

eval_batch_size = 1
rel_filter_threshold = 0.6
max_span_size = 6

store_predictions = true
store_examples = false
log_path = ./data/log
```

The evaluation reports:

- NER precision, recall, and F1;
- relation extraction without named entity classification;
- strict end-to-end relation extraction with named entity classification.

Strict relation extraction requires correct head entity span, tail entity span, entity types, relation type, and relation direction.

## Prediction

Use the following command to run prediction:

```bash
python run_ader.py predict --config configs/predict.conf
```

Example configuration:

```ini
model_type = ader
model_path = ./data/save/ADER/model_valid_best
tokenizer_path = ./data/models/BioLinkBERT

dataset_path = ./data/datasets/predict-example.json
types_path = ./data/datasets/types-example.json
predictions_path = ./data/datasets/predict-result.json

spacy_model = en_core_web_sm
eval_batch_size = 1
rel_filter_threshold = 0.6
max_span_size = 6
```

The prediction output is saved as a JSON file containing predicted entities and relations.

## Notes

- The AD-specific dataset used in the study is not provided in this repository.
- The pretrained encoder weights are not included.
- The example JSON files are provided only to illustrate the expected data format.
- The repository is intended to release the ADER model architecture and code framework.
- Users should prepare their own datasets and pretrained encoders before training or evaluation.

## Acknowledgement

This repository is developed based on a SpERT-style span-based joint entity and relation extraction framework. We gratefully acknowledge the original SpERT implementation by Markus Eberts and Adrian Ulges.

The ADER-specific modifications include:

- AD-oriented entity and relation extraction configuration;
- BioLinkBERT-based biomedical encoder setup;
- decoupled head-tail cross-attention for role-specific relation modeling;
- biaffine relation classification for ordered biomedical relations;
- training, evaluation, and prediction pipeline adaptation for AD literature extraction.

## Citation

If you use this code, please cite the ADER paper:

```bibtex
@article{ADER2026,
  title={ADER: A Model for Joint Entity and Directed Relation Extraction from Alzheimer’s Disease Literature},
  author={Shi, Jiangcheng and Su, Jinhao and Wang, Lijun and Liu, Lixue},
  journal={},
  year={2026}
}
```

Please also cite the original SpERT paper if you use the span-based extraction framework:

```bibtex
@inproceedings{eberts2020spert,
  title={Span-based Joint Entity and Relation Extraction with Transformer Pre-training},
  author={Eberts, Markus and Ulges, Adrian},
  booktitle={Proceedings of the 24th European Conference on Artificial Intelligence},
  year={2020}
}
```

## License

This project is released under the MIT License. The original SpERT copyright notice is retained, and ADER-specific modifications are additionally acknowledged in the `LICENSE` and `NOTICE.md` files.
