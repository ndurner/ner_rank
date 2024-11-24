# NER Ranking Harness

A collection of tools to evaluate and compare different Named Entity Recognition (NER) approaches, with a focus on German language and organization names.

## Overview

This repository contains:

- A ranking harness to evaluate different NER approaches
- A custom Presidio plugin for GLiNER integration
- Configuration for spaCy to include organization name recognition

For background and detailed results, see the accompanying [blog post](https://ndurner.github.io/ner).

## Files

- `ner_rank.py`: The main ranking harness
- `spacy_gliner_nlp_engine.py`: Presidio plugin for GLiNER integration
- `language-config.yml`: spaCy configuration with organization name recognition enabled

## Usage

```bash
python ner_rank.py --backends presidio_spacy gliner llm --llm_model meta-llama/Llama-3.2-3B-Instruct
```

### Limitations
This is not intended to be used as-is, but serve as a starting point for own experiments. E.g. the ranking mechanism does not penalize over-redaction and the configuration mechanism works by commenting code.

## Requirements

- Python 3.x
- spaCy
- Presidio
- GLiNER
- Transformers
- PyTorch
