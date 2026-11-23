"""Naive sequential pipeline, used ONLY as the baseline for the timing comparison.

Same work as the Airflow DAGs, same functions — but one shard after another, no cache, eval one example at a time.
Run it in the same environment as Airflow (inside the container) so the comparison is fair:

    docker compose exec airflow python -m baseline.naive_pipeline --runs 3

Raw downloads are excluded from timing (ingest skips files that already exist, in both pipelines).
"""
import argparse
import time

from src.etl import build_pairs, chunk, clean, dedupe, ingest, parse, schema, split
from src.eval import infer, scorer


def pipeline(eval_model, eval_prompt, eval_limit):
    shards = ingest.run()
    recs = []
    for s in shards:
        recs += parse.parse_shard(s)
    raw = len(recs)
    recs, _ = clean.clean_records(recs)
    cleaned = len(recs)
    recs, _, _ = dedupe.dedupe(recs, ingest.load_test_rows())
    pairs = [build_pairs.to_pair(chunk.chunk_record(r)) for r in recs]
    good, _ = schema.validate(pairs)
    train, dev = split.split(good)
    split.write(train, dev, ingest.DATA / "naive")

    examples = infer.load_examples("pqa_labeled", eval_limit)
    rows = infer.run(eval_model, None, eval_prompt, examples, batch_size=1, pred_root=None)
    m = scorer.score([r["gold"] for r in rows], [r["output"] for r in rows])
    print(f"funnel raw={raw} cleaned={cleaned} final={len(good)} | eval {m}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--eval-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--eval-prompt", default="structured")
    ap.add_argument("--eval-limit", type=int, default=100)
    ap.add_argument("--out", default="results/timing_naive.csv")
    a = ap.parse_args()
    for run in range(1, a.runs + 1):
        t = time.perf_counter()
        pipeline(a.eval_model, a.eval_prompt, a.eval_limit)
        secs = round(time.perf_counter() - t, 1)
        print(f"run {run}: {secs}s")
        infer.append_csv(a.out, {"pipeline": "naive", "run": run, "seconds": secs,
                                 "eval_model": a.eval_model, "eval_limit": a.eval_limit})


if __name__ == "__main__":
    main()
