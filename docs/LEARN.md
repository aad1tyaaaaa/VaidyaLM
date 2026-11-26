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

---

## Phase 3 · PR 8 — The ETL DAG (`preprocess_corpus`)

**What landed:** `dags/preprocess_corpus.py`, `src/etl/stages.py`, `Dockerfile`, `requirements-airflow.txt`,
`.env.example`, compose updates.

### Read the DAG bottom-up
```python
parsed  = parse_shard.expand(shard=ingest_raw())     # 1 task per shard, run in parallel
cleaned = clean_shard.expand(parsed=parsed)          # 1 task per parsed shard
split_and_push(validate_schema(build_instruction_pairs(chunk(dedupe_minhash(cleaned)))))
```
- **TaskFlow API (`@task`)**: plain Python functions become Airflow tasks; return values are passed downstream via
  **XCom** (Airflow's small key-value store). Rule: pass **paths and counts** through XCom, never datasets — XCom lives
  in Postgres and isn't built for megabytes.
- **Dynamic task mapping (`.expand`)**: the number of shards is only known at runtime (after ingest). `.expand()` creates
  one task instance per item — this is where the parallel speedup comes from. `max_active_tis_per_dagrun=8` caps it so
  your laptop doesn't run out of RAM.
- **Fan-in**: `dedupe_minhash(cleaned)` receives the *list* of all mapped results — it must, because dedupe is global.
- **Imports inside tasks**: the scheduler re-parses DAG files constantly; heavy top-level imports slow everything.

### The cache (`stages.py`)
Each output file `x.parquet` gets a sidecar `x.hash` = sha256(**input bytes + the stage's source code**).
- Same input and same code → skip (warm run).
- Change a cleaning rule in `clean.py` → hash changes → that stage re-runs automatically. Including the code in the key
  prevents the classic "stale cache" bug.
- Per-file sidecars instead of one shared `manifest.json`: 8 parallel tasks writing one JSON file = lost updates
  (race condition). Separate files = no locking needed.

### Pandas 3 gotcha (found while writing this)
Reading parquet in pandas 3 returns missing strings as `NaN`, and `bool(NaN)` is **True** — so `if r["label"]:` would
treat MedQuAD rows as labeled. `_read()` converts NaN back to `None` in one place.

### Docker changes
- Custom image (`Dockerfile`): installs CPU-only torch + our deps **once at build time**. The
  `_PIP_ADDITIONAL_REQUIREMENTS` trick from PR 2 would reinstall them on every start.
- Pinning `apache-airflow==2.10.5` in the same `pip install` stops pip from "helpfully" upgrading Airflow.
- Whole repo mounted at `/opt/airflow/repo` → `dags/`, `src/`, `baseline/`, `data/`, `results/` all visible, and
  the naive pipeline can run in the *same* container (fair timing).
- `env_file: .env` (optional) passes your `HF_TOKEN` in without committing it.
- `DAGS_ARE_PAUSED_AT_CREATION: false` + basic-auth API → the timing script (PR 11) can trigger DAGs over HTTP.

### ⏰ Reminders — run the ETL (Phase 3 exit criterion)
- [ ] `copy .env.example .env` and fill in `HF_TOKEN` + `HF_DATASET_REPO` (e.g. `yourname/vaidyalm-instruct`).
- [ ] `docker compose up -d --build` (first build ~5–10 min: torch is big).
- [ ] Airflow UI → `preprocess_corpus` → ▶ Trigger. Watch the **Grid** view: `parse_shard` shows `[N]` mapped tasks.
- [ ] Check `results/funnel.csv` and `data/quarantine.jsonl`. Dataset should appear on huggingface.co.
- [ ] Trigger it **again** → `parse_shard`/`clean_shard` finish in seconds (cache). That's your warm-cache run.
- [ ] Open a PR `phase-3/funnel-results` with `results/funnel.csv` and the funnel table pasted into PLAN.MD.
- [ ] Optional synthetic data: set `SYNTH_N=1000` in `.env` (uses HF Inference credits), rebuild, retrigger.

---

## Phase 4 · PR 9 — QLoRA fine-tuning

**What landed:** `src/train/sft_lora.py`, `src/train/configs/*.yaml`, `notebooks/train_kaggle.ipynb`.

### QLoRA in three ideas
1. **Quantization (the Q).** Base weights are stored in 4-bit **NF4** (a 4-bit format designed for normally distributed
   weights). 1.5B params × 0.5 byte ≈ 0.8 GB instead of 3 GB, so it fits a 16 GB T4 with room for activations.
   Double quantization also compresses the scaling constants.
2. **LoRA.** Instead of updating a big weight matrix W (d×d), learn two thin ones, A (d×r) and B (r×d), and use W + BA.
   With r=16 that's ~0.5% of the parameters. Only A and B get gradients; the 4-bit base is frozen.
   `lora_alpha` scales the update (alpha/r = 2 here). We adapt the attention projections `q,k,v,o_proj`.
3. **Memory tricks.** *Gradient checkpointing* recomputes activations in the backward pass instead of storing them
   (slower, much less memory). *Gradient accumulation* runs 4 micro-batches of 4 before one optimizer step, so the
   effective batch is 16.

### Why plain `transformers.Trainer` and not TRL's `SFTTrainer`
TRL's argument names change between versions, which breaks notebooks. The only thing we needed from it was
"loss on the response only", and `encode()` does that in about 8 lines:
- the prompt tokens get label **-100** (PyTorch's "ignore" value), so the model is graded only on its answer;
- prompt and response are tokenized separately and concatenated, so the mask lines up exactly (tested with the
  Qwen tokenizer: the loss covers `Answer: yes\nReason: ...<|im_end|>`, and the model also learns *when to stop*).

### Configs and ablations
`DEFAULTS` in `sft_lora.py` hold the PLAN.MD hyperparameters. Each YAML file lists only what it changes:
| Config | Question it answers |
|---|---|
| `r8`, `r16`, `r32` | Does LoRA capacity matter? |
| `r16_medquad` | Does MedQuAD alone (no yes/no data) help PubMedQA? |
| `r16_medquad_artificial` | What does synthetic data add on top? |

### Surviving Kaggle disconnects
`hub_strategy="checkpoint"` pushes the latest checkpoint to your HF repo every `save_steps`. On restart the script
looks for a local checkpoint first, then downloads `last-checkpoint/` from the Hub and resumes from it.

### ⏰ Reminders — train (Phase 4 exit criterion: beat the baseline on dev)
- [ ] The dataset must be on the HF Hub first (PR 8 checklist).
- [ ] Kaggle → import `notebooks/train_kaggle.ipynb` → Add-ons → Secrets → `HF_TOKEN` → set `HF_USER` → Run All.
- [ ] Run the smoke cell (200 rows) before the full run. Out of memory → `batch_size: 2`, `grad_accum: 8`.
- [ ] Do `r16` first. Compare `results/dev.csv`: does the adapter beat the base model? Then run the ablations one per session.
- [ ] Download `results.zip` → commit the CSVs + `train_logs/` in a PR `phase-4/results`; copy `predictions/` locally (it's git-ignored).

---

## Phase 5 · PR 10 — Eval DAG (`benchmark_models`) + report

**What landed:** `dags/benchmark_models.py`, `src/eval/report.py`.

### The DAG
```
list_models -> shard_eval_set -> batched_inference.expand(model x prompt x shard) -> score_and_report -> log_to_mlflow
```
- **DAG params**: trigger with a JSON config (Airflow UI → ▶ Trigger DAG w/ config) to pick models, prompts, limit.
  No code edits needed to add an adapter.
- **Cartesian fan-out**: 2 models × 3 prompts × 4 shards = 24 mapped inference tasks. `max_active_tis_per_dagrun=2`
  because each task may load a model.
- **Cache reuse**: `infer.run()` (PR 5) skips cached ids. So **GPU-heavy inference can happen on Kaggle**; you copy
  `results/predictions/` into the repo and the DAG only scores, reports and logs. One trigger, full table.

### Reading a confusion matrix
Rows are the true label, columns are what the model said. From the CPU test of the 0.5B model:
```
gold \ pred | yes | no | maybe
yes         |  0  |  0  |  1
no          |  0  |  0  |  1
maybe       |  0  |  0  |  1
```
Everything sits in the `maybe` column, so the model always hedges. Accuracy alone (33%) hides that; the matrix shows
it immediately. After fine-tuning you want the mass on the diagonal.

### MLflow
One MLflow run per (model, prompt) with params and metrics. For adapters we trained, the training loss curve from
`results/train_logs/<run>.json` is attached to the run, so the MLflow UI shows *how it trained* next to
*how it scored*.

### ⏰ Reminders — Phase 5 exit criterion: one trigger produces the full table
- [ ] Copy the Kaggle `predictions/` folders (baseline + adapters) into `results/predictions/`.
- [ ] Trigger `benchmark_models` with config:
      `{"models": [{"model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": null}, {"model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": "YOU/vaidyalm-qwen1.5b-lora-r16"}]}`
- [ ] Open `results/benchmark.md` and http://localhost:5000. Commit `benchmark.md`/`.csv` in a PR `phase-5/benchmark-results`.

---

## Phase 6 · PR 11 — Timing, README, demo

**What landed:** `baseline/time_airflow.py`, `demo/`, the full `README.md`.

### Timing the Airflow pipeline
`time_airflow.py` triggers both DAGs through Airflow's **REST API** (`POST /dags/{id}/dagRuns`), polls until each run
finishes, and uses Airflow's own `start_date`/`end_date`, not a stopwatch on your laptop.
- **cold**: deletes `data/interim` (ETL cache) and that model's cached predictions first, so everything is computed.
- **warm**: keeps them, which shows what caching buys on reruns.
Report cold and warm **separately**. Comparing only warm runs to the naive pipeline would overstate the gain.

Computing the result: `reduction % = (naive_mean - airflow_mean) / naive_mean × 100`, and say which (cold/warm).

### The README is the product
Recruiters read the README, not the code. It needs: what it is (one line), an architecture diagram (mermaid renders on
GitHub), how to run, **measured** results, and limitations. Listing limitations makes the numbers more believable.

### Demo
`demo/app.py` is a Gradio interface: paste an abstract and a question to get `Answer: yes|no|maybe` + a reason. It
imports the same `structured_text` prompt used in training and eval.

### ⏰ Reminders — final measurements (Phase 6)
- [ ] Find the Airflow password: `docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt`
- [ ] PowerShell: `$env:AIRFLOW_PASSWORD="..."` then
      `python -m baseline.time_airflow --mode cold --runs 3` and `--mode warm --runs 3`.
- [ ] Fill the README + PLAN.MD tables from `results/*.csv`. Compute the % reductions and say whether each one is relative or absolute.
- [ ] Optional: HF Spaces → New Space (Gradio) → upload `demo/app.py`, `demo/requirements.txt`, `src/eval/prompts.py`
      (as `prompts.py`) → Settings → Variables: `ADAPTER=YOU/vaidyalm-qwen1.5b-lora-r16`.
- [ ] Update the resume bullets in PLAN.MD §9 with the real X% and Y%.

## Maintenance — eval tests (`tests/test_eval.py`)

**What landed:** `tests/test_eval.py`, edge-case checks for the prompt templates and the scorer. Run it with
`python -m tests.test_eval` (needs `scikit-learn`).

### Why test the scorer so carefully
Every number in the README (accuracy, macro-F1, the ≥15% gain) comes from `scorer.py`. If the parser quietly misreads
model output, the headline result is wrong and nothing crashes to warn you. So the tests pin down the tricky cases:
- `"Nobody knows"` must **not** parse as `no`, and `"yesterday"` must **not** parse as `yes`. The regex uses word
  boundaries (`\b`) so a label only matches as a whole word.
- An explicit `Answer: yes` beats a stray `no` earlier in the text.
- Unparseable outputs count as **wrong** and are reported as `parse_fail_pct`, so a model can't look better by refusing to answer.
- If a class never appears in the gold labels, macro-F1 should not crash. It averages over all three labels and counts a missing one as 0.

It also checks one invariant: the prompt used for **training** (`format_user`) is identical to the **eval** structured
prompt. If those drift apart, the fine-tuned model is tested on a format it never saw.

### Manual to-do for you
- [ ] `pip install -r requirements.txt` in your local env, then run `python -m tests.test_eval` and `python -m tests.test_etl`.

## Maintenance — ETL cache tests (`tests/test_stages.py`)

**What landed:** `tests/test_stages.py`, checks for the caching code in `src/etl/stages.py`. Run it with
`python -m tests.test_stages`.

### The idea: a cache you can trust
The Airflow ETL skips work it already did. Each output parquet has a small sidecar file (`x.hash`) holding a
fingerprint (a SHA-256 hash) of the input bytes plus the stage code. Next run: same fingerprint → skip, different
fingerprint → recompute. This is what makes "warm" runs faster than "cold" ones, and that difference is a headline
number in the README.

A cache can fail in two ways, and only one of them is loud:
- **Too eager** (skips when it should recompute): you get stale data and no error. The tests change a file's content,
  and also just rename it, and check the fingerprint changes both times.
- **Too cautious** (always recomputes): the pipeline works but the timing gain is fake-small. The tests check that an
  unchanged key really skips the compute function.

Also checked: if the sidecar is missing, the parquet is not trusted and is rebuilt, and missing text values read back as
`None` instead of `NaN` (in pandas, `NaN` counts as true, so unlabeled rows would look labeled).

### Manual to-do for you
- [ ] Run `python -m tests.test_stages` locally (same requirements as the other tests).
- [ ] When you do the cold + warm Airflow runs, check the task logs show `cache_hit: True` on the warm run.

## Maintenance — report tests (`tests/test_report.py`)

**What landed:** `tests/test_report.py`, checks for `src/eval/report.py`, the script that turns cached model
predictions into `results/benchmark.md` / `benchmark.csv`. Run it with `python -m tests.test_report`.

### The idea: test the last step before the headline
`report.py` is the final hop between raw predictions and the table you will quote on your resume. It reads each
`model__prompt` folder of cached predictions, scores them, and draws a confusion matrix (rows = true answer,
columns = what the model said). The tests feed it a tiny fake prediction set where we know the right answer by hand:
- one correct, one wrong, one unparseable output → accuracy must be 1/3 and parse-fail 33.3%;
- the unparseable output must land in the `invalid` column of the matrix, not silently vanish;
- a model folder with no predictions must be skipped instead of crashing the report;
- passing `ids=[...]` must score only those examples (used to compare models on the same subset).

Tests write into a temporary folder, so they never touch your real `results/` files.

### Manual to-do for you
- [ ] Run `python -m tests.test_report` locally (needs `pandas` and `scikit-learn` from `requirements.txt`).
