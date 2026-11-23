# VaidyaLM — Progress

_Last updated: 2026-11-23_

Legend: ✅ done & tested · 🧪 code written, not yet run for real · ⏳ your turn (manual step) · ⬜ not started

## Snapshot
| Phase | Code | Runs / results | PRs |
|---|---|---|---|
| 0 · Setup | ✅ | ⏳ Docker/WSL2, HF token, Kaggle phone verification | #1, #2 |
| 1 · Baseline eval | ✅ | ⏳ run `notebooks/baseline_kaggle.ipynb` | #3, #4, #5 |
| 2 · Naive pipeline | ✅ | ⏳ 3 timed runs → `results/timing_naive.csv` | #7 |
| 3 · ETL DAG | ✅ modules · 🧪 DAG | ⏳ trigger in Airflow, push dataset to HF | #6, #8 |
| 4 · Fine-tuning | 🧪 | ⏳ Kaggle training r16 + ablations | #9 (draft) |
| 5 · Eval DAG | 🧪 | ⏳ one trigger → `results/benchmark.md` | #10 (draft) |
| 6 · Measure + write-up | 🧪 | ⏳ timing runs, fill README tables, demo | #11 (draft) |

## PRs and merge order
All PRs are **stacked**: each is based on the one before it. Merge in order: **#1 → #2 → … → #11**. When you merge one
and delete its branch, GitHub retargets the next PR to `main` automatically.
#9–#11 are **drafts**: the code is complete but hasn't run on real hardware yet. Mark each one "Ready for review" once
its checklist in `docs/LEARN.md` passes. This PR (`PROGRESS.md`) is independent and can be merged any time.

## What's been verified
- `python -m src.eval.scorer`: the parser and metrics pass their asserts.
- `python -m tests.test_etl`: parse → clean → dedupe (dup + leakage) → chunk → pairs → schema → split pass end to end.
- Cached batched inference on CPU (Qwen2.5-0.5B): reruns only generate the missing ids.
- Real MedQuAD through the ETL: **47,441 raw → 16,390 cleaned → 15,330 deduped** (31,034 empty answers in the
  copyright-stripped MedlinePlus collections; 1,060 near-duplicates).
- Training label masking checked with the real Qwen tokenizer: loss only on `Answer/Reason` + end token.
- `report.py` renders the table + confusion matrix from real cached predictions.

## Bugs found and fixed along the way
- MedQuAD reuses document ids across files, so pair ids collided and crashed the dedupe. Ids now come from file names.
- In pandas 3, missing strings read from parquet come back as `NaN` (truthy), which made unlabeled rows look labeled. Fixed with a shared `_read()`.

## Not yet verified (needs your hardware or accounts)
- Docker build + both DAGs actually running in Airflow (no Docker on the dev machine yet).
- A cold + warm run of the cached DAG stages on real data (the next thing I'd run locally).
- Anything needing a GPU: the baseline on 1,000 examples and QLoRA training.
- HF Hub dataset/adapter pushes, synthetic labeling via HF Inference.

## Next up — your checklist (details in `docs/LEARN.md`)
1. ⏳ Review & merge #1 → #8.
2. ⏳ Kaggle phone verification + HF write token → run the baseline notebook → PR `phase-1/baseline-results`.
3. ⏳ Install WSL2 + Docker Desktop → `docker compose up -d --build` → trigger `preprocess_corpus`.
4. ⏳ Naive timing runs inside the container.
5. ⏳ Kaggle training (`r16` first), then the ablations.
6. ⏳ `benchmark_models` + timing runs → fill the README results → resume bullets.

## Results (fill in as measured)
| Metric | Value |
|---|---|
| Baseline best (prompt, accuracy / macro-F1) | – |
| Fine-tuned r16 structured (accuracy / macro-F1) | – |
| Relative accuracy gain | – (target ≥ 15%) |
| Naive vs Airflow cold / warm, mean seconds | – / – / – |
| Wall-clock reduction | – (target ≥ 25%) |
