# VaidyaLM

Health-domain LLM: QLoRA fine-tune of Qwen2.5-1.5B-Instruct on medical QA, an Airflow ETL that turns
unstructured medical text into instruction data, and an automated PubMedQA benchmark.

- Plan: [PLAN.MD](PLAN.MD)
- Learning notes per phase: [docs/LEARN.md](docs/LEARN.md)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Results go in `results/` and are filled in as each phase lands.
