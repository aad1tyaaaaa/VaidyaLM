"""File-based, cached wrappers around the pure ETL functions. Used by the Airflow DAG.

Cache: each output parquet gets a sidecar `<file>.hash` = sha256(input bytes + stage code).
Same hash next run -> skip. Sidecars (not one shared manifest) so parallel tasks never write the same file.
"""
import hashlib
import json
import pathlib

import pandas as pd

from src.etl import build_pairs, chunk, clean, dedupe, ingest, parse, schema, split, synth

INTERIM = ingest.DATA / "interim"
FINAL = ingest.DATA / "final"


def _hash(*paths):
    h = hashlib.sha256()
    for p in map(pathlib.Path, paths):
        for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
            if f.is_file():
                h.update(f.name.encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def _read(path):
    """Parquet -> records with real None for missing values (pandas 3 would give NaN, which is truthy)."""
    df = pd.read_parquet(path).astype(object)
    return df.where(df.notna(), None).to_dict("records")


def _cached(out, key, compute):
    """Returns (records_count, from_cache)."""
    side = out.with_suffix(".hash")
    if out.exists() and side.exists() and side.read_text() == key:
        return len(pd.read_parquet(out, columns=["id"])), True
    recs = compute()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs, columns=parse.COLUMNS).to_parquet(out, index=False)
    side.write_text(key)
    return len(recs), False


def parse_stage(shard):
    out = INTERIM / "parsed" / f"{pathlib.Path(shard).stem}.parquet"
    n, hit = _cached(out, _hash(shard, parse.__file__, synth.__file__), lambda: parse.parse_shard(shard))
    return {"path": str(out), "n": n, "cache_hit": hit}


def clean_stage(parsed):
    src = pathlib.Path(parsed["path"])
    out = INTERIM / "cleaned" / src.name

    def compute():
        kept, reasons = clean.clean_records(_read(src))
        (out.parent / f"{src.stem}.reasons.json").write_text(json.dumps(reasons))
        return kept

    out.parent.mkdir(parents=True, exist_ok=True)
    n, hit = _cached(out, _hash(src, clean.__file__), compute)
    return {"path": str(out), "n_in": parsed["n"], "n": n, "cache_hit": hit}


def dedupe_stage(cleaned):
    recs = [r for c in cleaned for r in _read(c["path"])]
    kept, dup, leak = dedupe.dedupe(recs, ingest.load_test_rows())
    out = INTERIM / "deduped.parquet"
    pd.DataFrame(kept, columns=parse.COLUMNS).to_parquet(out, index=False)
    return {"path": str(out), "raw": sum(c["n_in"] for c in cleaned), "cleaned": len(recs),
            "duplicates": dup, "leaked": leak, "deduped": len(kept)}


def chunk_stage(d):
    recs = [chunk.chunk_record(r) for r in _read(d["path"])]
    out = INTERIM / "chunked.parquet"
    pd.DataFrame(recs, columns=parse.COLUMNS).to_parquet(out, index=False)
    return {**d, "path": str(out)}


def build_pairs_stage(d):
    pairs = [build_pairs.to_pair(r) for r in _read(d["path"])]
    out = INTERIM / "pairs.parquet"
    pd.DataFrame(pairs, columns=split.COLUMNS).to_parquet(out, index=False)
    return {**d, "path": str(out), "pairs": len(pairs)}


def validate_stage(d):
    pairs = _read(d["path"])
    good, bad = schema.validate(pairs)
    with open(ingest.DATA / "quarantine.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(b) + "\n" for b in bad)
    out = INTERIM / "valid.parquet"
    pd.DataFrame(good, columns=split.COLUMNS).to_parquet(out, index=False)
    return {**d, "path": str(out), "valid": len(good), "quarantined": len(bad)}


def split_and_push_stage(d, repo=None, results_dir="results"):
    good = _read(d["path"])
    train, dev = split.split(good)
    split.write(train, dev, FINAL)
    funnel = {k: d[k] for k in ("raw", "cleaned", "duplicates", "leaked", "deduped", "valid", "quarantined")}
    funnel |= {"train": len(train), "dev": len(dev)}
    sources = pd.Series([p["source"] for p in train]).value_counts().to_dict()
    card = split.dataset_card(funnel, sources)
    (FINAL / "README.md").write_text(card, encoding="utf-8")
    pathlib.Path(results_dir).mkdir(exist_ok=True)
    pd.DataFrame(funnel.items(), columns=["stage", "rows"]).to_csv(f"{results_dir}/funnel.csv", index=False)
    if repo:
        split.push(FINAL, repo, card)
    return funnel
