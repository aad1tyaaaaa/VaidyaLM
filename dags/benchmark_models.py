"""Eval DAG: base model + LoRA adapters x prompts on PubMedQA pqa_labeled.

list_models -> shard_eval_set -> batched_inference.expand(model x prompt x shard)
            -> score_and_report -> log_to_mlflow

Predictions are cached per (model, prompt, example_id) in results/predictions/, so reruns only compute what's
missing. No GPU locally? Run notebooks/baseline_kaggle.ipynb (or the eval cells of train_kaggle.ipynb), copy the
results/predictions/ folders it produces into this repo, then trigger: inference is skipped for cached ids.
"""
import json
from datetime import datetime

from airflow.decorators import dag, task
from airflow.models.param import Param

EXAMPLES = "data/interim/eval_examples.json"


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vaidyalm"],
    params={
        "models": Param(
            [{"model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": None}],
            type="array",
            description='e.g. [{"model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": "you/vaidyalm-qwen1.5b-lora-r16"}]',
        ),
        "prompts": Param(["zero_shot", "few_shot", "structured"], type="array"),
        "limit": Param(None, type=["null", "integer"], description="first N test examples (None = all 1000)"),
        "n_shards": Param(4, type="integer"),
        "batch_size": Param(16, type="integer"),
    },
)
def benchmark_models():
    @task
    def list_models(params=None):
        return [{"model": m["model"], "adapter": m.get("adapter"), "prompt": p}
                for m in params["models"] for p in params["prompts"]]

    @task
    def shard_eval_set(jobs, params=None):
        import pathlib

        from src.eval import infer

        examples = infer.load_examples("pqa_labeled", params["limit"])
        pathlib.Path(EXAMPLES).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(EXAMPLES).write_text(json.dumps(examples))
        n = params["n_shards"]
        return [{**j, "shard": i, "n_shards": n, "batch_size": params["batch_size"]} for j in jobs for i in range(n)]

    @task(max_active_tis_per_dagrun=2)  # each one may load a model; keep RAM in check
    def batched_inference(job):
        import pathlib

        from src.eval import infer

        examples = json.loads(pathlib.Path(EXAMPLES).read_text())[job["shard"]::job["n_shards"]]
        rows = infer.run(job["model"], job["adapter"], job["prompt"], examples, job["batch_size"])
        return {"key": f"{infer.model_name(job['model'], job['adapter'])}__{job['prompt']}", "n": len(rows)}

    @task
    def score_and_report(done):
        import pathlib

        from src.eval import report

        keys = sorted({d["key"] for d in done})
        ids = [e["id"] for e in json.loads(pathlib.Path(EXAMPLES).read_text())]
        return report.build(keys, ids)

    @task
    def log_to_mlflow(rows):
        import os
        import pathlib

        import mlflow

        mlflow.set_experiment("vaidyalm-benchmark")
        for r in rows:
            with mlflow.start_run(run_name=f"{r['model']}__{r['prompt']}"):
                mlflow.log_params({"model": r["model"], "prompt": r["prompt"], "n": r["n"]})
                mlflow.log_metrics({k: r[k] for k in ("accuracy", "macro_f1", "parse_fail_pct")})
                # training curve, if this model is an adapter we trained (results/train_logs/<run_name>.json)
                run_name = r["model"].split("lora-")[-1] if "lora-" in r["model"] else None
                log = pathlib.Path(f"results/train_logs/{run_name}.json")
                if run_name and log.exists():
                    for e in json.loads(log.read_text())["log_history"]:
                        if "loss" in e:
                            mlflow.log_metric("train_loss", e["loss"], step=e["step"])
                        if "eval_loss" in e:
                            mlflow.log_metric("eval_loss", e["eval_loss"], step=e["step"])
                mlflow.log_artifact("results/benchmark.md")
        print("logged", len(rows), "runs to", os.environ.get("MLFLOW_TRACKING_URI"))

    jobs = list_models()
    done = batched_inference.expand(job=shard_eval_set(jobs))
    log_to_mlflow(score_and_report(done))


benchmark_models()
