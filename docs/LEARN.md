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
