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
