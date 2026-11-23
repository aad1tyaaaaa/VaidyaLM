"""Download raw sources once and split them into shards. Returns shard paths.

Shards:
  data/raw/MedQuAD-master/<collection>/   (one dir of XML files per MedQuAD collection)
  data/raw/pubmedqa/artificial_NNN.jsonl   (ARTIFICIAL_N rows of pqa_artificial)
  data/raw/pubmedqa/synthetic_NNN.jsonl    (SYNTH_N unlabeled abstracts to label with a larger model; 0 = off)
  data/raw/pubmedqa/test_labeled.jsonl     (pqa_labeled, ONLY used to filter leakage, never trained on)
"""
import io
import json
import os
import pathlib
import urllib.request
import zipfile

DATA = pathlib.Path(os.environ.get("VAIDYA_DATA", "data"))
RAW = DATA / "raw"
MEDQUAD_ZIP = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
SHARD_ROWS = 5000


def _write_shards(rows, prefix, size):
    out = RAW / "pubmedqa"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(rows), size):
        with open(out / f"{prefix}_{i // size:03d}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows[i : i + size])


def _pubmedqa(config, n):
    from datasets import load_dataset

    from src.eval.prompts import from_hf

    split = f"train[:{n}]" if n else "train"
    return [from_hf(e) | {"long_answer": e.get("long_answer", "")} for e in load_dataset("qiaojin/PubMedQA", config, split=split)]


def _unlabeled(n):
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split=f"train[:{n}]")
    return [{"id": str(e["pubid"]), "question": e["question"], "context": "\n".join(e["context"]["contexts"])} for e in ds]


def run(artificial_n=None, synth_n=None):
    artificial_n = int(artificial_n if artificial_n is not None else os.environ.get("ARTIFICIAL_N", 20000))
    synth_n = int(synth_n if synth_n is not None else os.environ.get("SYNTH_N", 0))
    pq = RAW / "pubmedqa"

    medquad = RAW / "MedQuAD-master"
    if not medquad.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(MEDQUAD_ZIP) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(RAW)
    if not (pq / "test_labeled.jsonl").exists():
        _write_jsonl(pq / "test_labeled.jsonl", _pubmedqa("pqa_labeled", None))
    if artificial_n and not list(pq.glob("artificial_*.jsonl")):
        _write_shards(_pubmedqa("pqa_artificial", artificial_n), "artificial", SHARD_ROWS)
    if synth_n and not list(pq.glob("synthetic_*.jsonl")):
        _write_shards(_unlabeled(synth_n), "synthetic", 250)  # small shards: each one is many API calls

    shards = sorted(str(d) for d in medquad.iterdir() if d.is_dir() and any(d.glob("*.xml")))
    shards += sorted(str(p) for p in pq.glob("artificial_*.jsonl"))
    if synth_n:
        shards += sorted(str(p) for p in pq.glob("synthetic_*.jsonl"))
    return shards


def load_test_rows():
    with open(RAW / "pubmedqa" / "test_labeled.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
