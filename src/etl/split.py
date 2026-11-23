"""Train/dev split, parquet output, optional push to the HF Hub with a dataset card."""
import os
import pathlib
import random

import pandas as pd

COLUMNS = ["id", "source", "question", "context", "response", "label"]


def split(pairs, n_dev=100, seed=0):
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    dev = {i for i in idx[:n_dev]}
    return [p for i, p in enumerate(pairs) if i not in dev], [p for i, p in enumerate(pairs) if i in dev]


def write(train, dev, out_dir):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train, columns=COLUMNS).to_parquet(out / "train.parquet", index=False)
    pd.DataFrame(dev, columns=COLUMNS).to_parquet(out / "dev.parquet", index=False)


def dataset_card(funnel, sources):
    rows = "\n".join(f"| {k} | {v:,} |" for k, v in funnel.items())
    mix = "\n".join(f"| {k} | {v:,} |" for k, v in sources.items())
    return f"""---
license: cc-by-4.0
task_categories: [question-answering]
language: [en]
tags: [medical, instruction-tuning]
---
# vaidyalm-instruct

Instruction pairs for medical QA fine-tuning, built by the VaidyaLM Airflow ETL
(parse -> clean -> MinHash dedupe -> leakage filter vs PubMedQA `pqa_labeled` -> chunk -> pydantic validation).

Fields: `id, source, question, context, response, label`. Labeled rows use the response format
`Answer: yes|no|maybe` + `Reason: ...`.

## Data funnel
| Stage | Rows |
|---|---|
{rows}

## Source mix (train)
| Source | Rows |
|---|---|
{mix}

## Sources and licenses
- MedQuAD (Ben Abacha & Demner-Fushman, 2019), CC BY 4.0. Collections with removed answers are dropped by the cleaner.
- PubMedQA `pqa_artificial` (Jin et al., 2019), MIT.
- Synthetic: labels for PubMedQA `pqa_unlabeled` from a larger model, kept only if self-consistent and grounded.

`pqa_labeled` is the held-out test set: near-duplicates of it are removed from this dataset.
"""


def push(out_dir, repo, card):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    out = pathlib.Path(out_dir)
    ds = load_dataset("parquet", data_files={"train": str(out / "train.parquet"), "dev": str(out / "dev.parquet")})
    ds.push_to_hub(repo, token=token)
    HfApi(token=token).upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
