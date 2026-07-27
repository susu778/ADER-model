# Notice

This repository contains the implementation of ADER, a joint entity and directed relation extraction model for Alzheimer’s disease literature.

The codebase is developed based on the SpERT-style span-based joint entity and relation extraction framework. We gratefully acknowledge the original SpERT implementation by Markus Eberts.

The ADER-specific modifications include:

- BioLinkBERT-based contextual encoding configuration;
- decoupled head-tail cross-attention for role-specific relation modeling;
- biaffine relation classification for ordered and direction-sensitive biomedical relations;
- AD-oriented entity and relation extraction configuration;
- ablation variants and prediction pipeline for biomedical knowledge graph construction.

The AD-specific annotated dataset used in our study is not included in this repository.