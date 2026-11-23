# VaidyaLM — Learning Notes

One section per PR. Read the section, then read the code it points to.

---

## Phase 0 · PR 1 — Repo setup

**What landed:** `.gitignore`, `requirements.txt`, `README.md`.

### Concepts
- **`.gitignore` first, always.** Datasets (`data/`), MLflow runs (`mlruns/`) and `.env` (your tokens) must never hit
  GitHub. Once a secret is pushed it's public forever, even if you delete it later — you'd have to rotate the token.
- **Why parquet is ignored:** data is *regenerable* from the ETL. Git tracks code; HF Hub tracks datasets/models.
  That split is the whole "Versioning" row in PLAN §2.
- **`requirements.txt` uses `>=` floors, not exact pins.** Kaggle/Colab ship their own torch + CUDA; hard pins would
  fight them. `bitsandbytes` is marked `sys_platform == "linux"` because 4-bit QLoRA needs CUDA — it only matters on
  the Kaggle GPU box, not your Windows laptop.
- **Virtual env (`.venv`)** keeps this project's packages out of your global Python.

### Try it
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "import pandas, datasets, sklearn; print('ok')"
```

### ⏰ Reminders — things only YOU can do
- [ ] **Hugging Face**: create account → Settings → Access Tokens → new token with **write** scope. Save it in a local
      `.env` as `HF_TOKEN=...` (already git-ignored). Needed in Phase 3 (push dataset) and Phase 4 (push adapters).
- [ ] **Kaggle**: create account → Settings → **verify phone number** (GPU is locked without it). Needed in Phase 1.
- [ ] **Docker Desktop** (+ WSL2 on Windows): not needed until Phase 3. See PR 2 notes.

---

## Phase 0 · PR 2 — Docker, Airflow, MLflow (code only — run it in Phase 3)

**What landed:** `docker-compose.yaml` with 3 services.

### The big picture
```
 you (browser) ──► :8080 Airflow ──runs──► dags/*.py ──imports──► src/etl/*.py
                         │                                        │
                         └── stores its own state in ► postgres   └─ logs metrics ► :5000 MLflow
```
- **Docker** runs each tool in an isolated box (container) so you don't install Airflow on Windows (it doesn't
  support Windows natively — that's why WSL2 is required).
- **docker compose** = a recipe for several containers that talk to each other by service name
  (`http://mlflow:5000` works *inside* the network; you use `localhost:5000` from your browser).
- **Airflow** is a scheduler: you write a **DAG** (Directed Acyclic Graph) of tasks in Python; Airflow runs them in
  dependency order, retries failures, shows logs per task, and — key for our timing goal — runs independent tasks
  **in parallel** (`LocalExecutor` + `.expand()` dynamic task mapping, Phase 3).
- **Postgres** is Airflow's memory (which task ran, when, did it succeed). The default SQLite can't do parallelism.
- **MLflow** is a lab notebook for experiments: every eval/training run logs params (model, prompt, LoRA rank) and
  metrics (accuracy, F1) so you can compare runs in a table instead of scrolling terminal output.

### Read the compose file line by line
- `volumes: ./dags:/opt/airflow/dags` — your local folder is *mounted* into the container. Edit a DAG in VS Code,
  Airflow sees it within ~30s. No rebuild.
- `PYTHONPATH: /opt/airflow/repo` — lets DAGs do `from src.etl import parse`.
- `_PIP_ADDITIONAL_REQUIREMENTS` — quick way to add packages at container start. Fine for dev; for prod you'd build
  a custom image (slow startups are the price).
- `healthcheck` + `condition: service_healthy` — Airflow waits until Postgres actually accepts connections.
- We use one `standalone` container instead of the official 8-service file. Less to break, same parallelism for one
  machine. Upgrade path is noted in the file.

### ⏰ Reminders — do this before Phase 3
- [ ] Install **WSL2**: PowerShell (admin) → `wsl --install` → reboot.
- [ ] Install **Docker Desktop** → Settings → Resources → give it **≥ 4 GB RAM** (Airflow is hungry).
- [ ] From the repo folder: `docker compose up -d`, wait ~1–2 min, then:
  - `docker compose logs airflow | findstr password` → the admin password
  - open http://localhost:8080 (Airflow) and http://localhost:5000 (MLflow). Both load = Phase 0 done ✅
- [ ] Stop everything: `docker compose down` (add `-v` to also wipe the Postgres volume).

---

## Phase 1 · PR 3 — Prompts + scorer

**What landed:** `src/eval/prompts.py`, `src/eval/scorer.py`.

### The task: PubMedQA
Each example = a research abstract (**context**) + a **question** → label **yes / no / maybe**. It's a 3-class
classification problem disguised as QA. `pqa_labeled` has 1,000 expert-labeled examples — our **test set, never
trained on** (if the model sees test data during training, the score is meaningless: that's *leakage*).

### Three prompts = three ways of asking
| Template | Idea | Why it matters |
|---|---|---|
| `zero_shot` | Just ask. | The floor — what the raw model does. |
| `few_shot` | Show 3 solved examples first (*in-context learning*). | Teaches the format without training. |
| `structured` | Force `Answer: yes|no|maybe` + one-line reason. | Easy to parse; we'll **train** on this same format in Phase 4. |

Separating "prompting gains" from "fine-tuning gains" is the point of the ablation table in Phase 6.

### Why the scorer is the most important file
LLMs output free text, not labels. `parse_label` turns `"**Answer:** Yes, because..."` into `"yes"`:
1. Look for `answer ... yes|no|maybe` first (structured output).
2. Fall back to the first standalone yes/no/maybe word.
3. Nothing found → `None` → counted as **wrong**, and tracked as **parse-fail %**.

Never silently drop unparseable outputs — that would inflate accuracy. Reporting parse-fail % is honest.

### Metrics
- **Accuracy** = correct / total. Misleading if classes are imbalanced (PubMedQA is ~55% yes).
- **Macro-F1** = F1 per class, then a plain average. A model that always says "yes" gets decent accuracy but terrible
  macro-F1 because "no" and "maybe" F1 are 0. That's why we report both.

### Try it
```bash
python -m src.eval.scorer      # runs the built-in asserts, prints "scorer ok"
```
`python -m src.eval.scorer` (not `python src/eval/scorer.py`) runs it as a module from the repo root so
`from src.eval import ...` imports work.

---

## Phase 1 · PR 4 — Baseline inference on Kaggle

**What landed:** `src/eval/infer.py`, `notebooks/baseline_kaggle.ipynb`.

### What `infer.py` does, step by step
1. **Load tokenizer with `padding_side="left"`.** Batched generation pads short prompts. For decoder-only models
   the padding must be on the *left*, so every prompt ends right where generation starts. Right padding = garbage output.
2. **Load model in `float16`.** The T4 GPU has no bfloat16; fp16 halves memory vs fp32 (1.5B params ≈ 3 GB).
3. **`apply_chat_template`.** Instruct models were trained on a specific chat format (`<|im_start|>user ...`).
   Feeding raw text skips that and quietly lowers accuracy.
4. **Greedy decoding (`do_sample=False`) + fixed seed.** Same input → same output. Benchmarks must be reproducible.
5. **Slice off the prompt:** `gen[:, input_len:]` — `generate` returns prompt + answer; we only score the answer.
6. **Save raw outputs** to `results/predictions/*.jsonl` (git-ignored) so you can *read* failures, and
   **append metrics** to `results/baseline.csv` — this is the number every later claim is compared against.
7. `--adapter` is already wired for Phase 4: same script evaluates the fine-tuned model → apples-to-apples.

### Why Kaggle
Your laptop has no CUDA GPU. Kaggle gives ~30 GPU-hours/week free. The notebook is a *thin wrapper*: clone repo →
run the script. All real logic lives in `src/` so it's versioned and reviewable — never bury logic in notebooks.

### ⏰ Reminders — run the baseline (this is the Phase 1 exit criterion)
- [ ] Merge PRs #1 → #2 → #3 → #4 in order (each is stacked on the previous).
- [ ] Kaggle → New Notebook → File → Import → upload `notebooks/baseline_kaggle.ipynb`.
- [ ] Settings: Accelerator **GPU T4**, Internet **On** (needs phone verification).
- [ ] Run the smoke-test cell first. Then the full cells (~20–40 min total).
- [ ] Download `baseline.csv`, put it in `results/`, open a PR `phase-1/baseline-results`, and fill in the table in
      PLAN.MD §6. **Report exactly what you measured.**
- [ ] Open a few lines of the `.jsonl` predictions — look at *how* the base model fails. That tells you what
      fine-tuning must fix.

---

## Phase 1 · PR 5 — Flat examples + prediction cache

**What changed:** `prompts.py` and `infer.py` were refactored so every later phase (training, ETL dev set,
Airflow eval DAG, naive pipeline) can reuse them.

### 1. One example format everywhere
PubMedQA on HF nests context as `{"contexts": [...]}`. Our ETL produces a flat string. Instead of two code paths,
`prompts.from_hf()` converts HF rows to `{"id", "question", "context", "label"}` once, at the edge. Rule of thumb:
**normalize data at the boundary, keep the core simple.**

### 2. `format_user()` — train/eval format match
Fine-tuning teaches the model to respond to a *specific* prompt shape. If you train on one format and evaluate on
another, you measure format confusion, not knowledge. `format_user()` is the single function the trainer uses, and it
calls the exact same `structured_text()` the evaluator uses.

### 3. Prediction cache
`results/predictions/<model>__<prompt>/part-*.jsonl`. Before generating, `run()` reads every cached id and only
generates the missing ones. Why it matters:
- Kaggle session dies at example 800? Rerun → only 200 left.
- Airflow eval DAG (Phase 5) gets "warm cache" speedups for free → part of the ≥25% timing goal.
- Each parallel shard writes its own `part-<hash>.jsonl` → no two processes ever write the same file (no locks needed).

### 4. Lazy imports
`torch`/`transformers` are imported *inside* functions. Airflow parses DAG files every ~30 s; importing torch at the
top would make that slow. Also lets `report.py` read the cache without loading a GPU stack.

### Real observation from testing
Qwen2.5-**0.5B** on CPU answered **"maybe"** to "Does smoking improve lung function?" given a context that clearly
says no. That's the failure mode fine-tuning targets: the base model hedges instead of committing to the context.

---

## Phase 3 · PR 6 — ETL building blocks (pure functions)

**What landed:** `src/etl/{ingest,parse,synth,clean,dedupe,chunk,build_pairs,schema,split}.py`, `tests/test_etl.py`.

> Why Phase 3 before Phase 2? The naive pipeline (Phase 2) must do the *same work* as the Airflow pipeline, or the
> timing comparison is fake. So the logic lives here once; Phase 2 calls it sequentially, Phase 3's DAG calls it in parallel.

### The flow — one record shape all the way through
```
ingest  -> shards (12 MedQuAD dirs + PubMedQA jsonl files)
parse   -> {id, source, question, context, answer, label}
clean   -> normalized text, bad rows dropped with a reason
dedupe  -> near-duplicates + anything resembling the TEST set removed
chunk   -> long text cut at sentence boundaries to fit 1024 tokens
build_pairs -> {..., response}   e.g. "Answer: yes\nReason: <first sentence>"
schema  -> pydantic validates; failures -> quarantine with reason
split   -> train / dev(100) parquet -> HF Hub + dataset card
```

### Concepts worth knowing
- **Pure functions vs. I/O.** Every module takes Python lists and returns Python lists. No files, no Airflow. That makes
  them testable in milliseconds (`tests/test_etl.py`) and reusable by both pipelines. File reading/caching comes in PR 8.
- **XML parsing (`parse.py`).** `lxml` with `recover=True` tolerates slightly broken files instead of crashing a
  47k-row job on one bad file. BeautifulSoup only runs when a string actually contains `<` (HTML) — cheap path first.
- **IDs must be unique — real bug found while testing.** MedQuAD reuses `Document id` values across files, so the
  first version produced duplicate ids and the dedupe step crashed. Fix: build ids from the *file name*. Lesson: always
  run your ETL on real data early; synthetic tests only catch the bugs you imagined.
- **Quality filters (`clean.py`).** Every drop has a named reason (`empty_answer`, `language`, ...). Counting reasons
  turns "the data got smaller" into an explainable **data funnel**.
- **MinHash LSH (`dedupe.py`).** Comparing every pair of 40k docs = 800M comparisons. MinHash compresses each doc into
  64 numbers such that similar docs get similar numbers; LSH buckets them so you only compare likely matches.
  Threshold 0.85 ≈ "85% of 3-word shingles shared".
- **Leakage filter.** The 1,000 test questions are inserted into the LSH *first*. Any training row that matches one is
  dropped and counted as `leaked`. This is what makes the final accuracy number trustworthy.
- **Synthetic data (`synth.py`).** A big model labels unlabeled abstracts. We only keep a label if (1) two samples at
  different temperatures agree (*self-consistency*) and (2) the reason uses words from the abstract (*grounding*).
  Cheap filters that remove a lot of hallucination. Off by default because API calls cost credits.
- **pydantic (`schema.py`).** Declarative rules (`min_length`, allowed labels, "response starts with its label").
  Invalid rows aren't silently dropped — they go to a quarantine file you can inspect.

### Real numbers (MedQuAD only, measured on this machine)
| Stage | Rows |
|---|---|
| raw | 47,441 |
| cleaned | 16,390 (31,034 empty answers: MedlinePlus collections had answers removed for copyright) |
| deduped | 15,330 (1,060 near-duplicates) |

Dedupe took ~44 s — it's the slowest step and the one that *can't* be parallelized (it's a global comparison).

### Try it
```bash
python -m tests.test_etl        # -> etl ok
```

---

## Phase 2 · PR 7 — Naive sequential pipeline (the timing baseline)

**What landed:** `baseline/naive_pipeline.py`.

### Why build a deliberately simple version?
The resume claim is "Airflow cut benchmarking time by Y%". A percentage needs a **baseline**. This script is that
baseline: the honest version a person would write first — a plain `for` loop.

### What makes it "naive" (and what doesn't)
| | Naive | Airflow (PR 8, PR 10) |
|---|---|---|
| Shards | one after another | in parallel (`.expand()`) |
| Re-running unchanged shards | re-parses everything | skips (content-hash cache) |
| Eval | `batch_size=1`, no cache | batched + prediction cache |
| ETL logic | **same functions** | **same functions** |

The rule from PLAN.MD: *realistic code that's simply unoptimized, not deliberately slowed down.* No `sleep()`, no
duplicated work. If the comparison were rigged, the number would be worthless in an interview.

### Fair-measurement rules
- Run both pipelines **in the same environment** (inside the Airflow container) — laptop-vs-Docker would compare
  machines, not pipelines.
- Raw downloads are excluded (both skip existing files) — network speed isn't what we're measuring.
- Same eval config for both: Qwen2.5-**0.5B**, `structured` prompt, 100 examples. Small enough for CPU.
- 3 runs, report the mean (and look at the spread — one slow run can be a cold disk cache).
- `time.perf_counter()` — monotonic, high-resolution; never use `time.time()` for durations.

### ⏰ Reminders — measure the naive baseline (after Docker is up, see PR 2/PR 8 notes)
- [ ] `docker compose exec airflow python -m baseline.naive_pipeline --runs 3`
- [ ] Result lands in `results/timing_naive.csv`. Commit it in a PR `phase-2/naive-timing`.
