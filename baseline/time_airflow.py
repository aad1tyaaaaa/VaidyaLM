"""Time the Airflow pipeline (preprocess_corpus + benchmark_models) via the REST API. Runs on your host.

    python -m baseline.time_airflow --mode cold --runs 3
    python -m baseline.time_airflow --mode warm --runs 3

Same eval config as baseline/naive_pipeline.py defaults so the numbers are comparable.
Duration = sum of each DAG run's (end_date - start_date) as recorded by Airflow.
cold = delete ETL cache + the eval key's cached predictions before each run; warm = keep them.
"""
import argparse
import base64
import json
import os
import pathlib
import shutil
import time
import urllib.request
from datetime import datetime

from src.eval import infer

API = os.environ.get("AIRFLOW_API", "http://localhost:8080/api/v1")


def call(method, path, body=None):
    auth = base64.b64encode(f"{os.environ.get('AIRFLOW_USER', 'admin')}:{os.environ['AIRFLOW_PASSWORD']}".encode()).decode()
    req = urllib.request.Request(f"{API}{path}", method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def run_dag(dag_id, conf):
    run = call("POST", f"/dags/{dag_id}/dagRuns", {"conf": conf})
    while run["state"] not in ("success", "failed"):
        time.sleep(5)
        run = call("GET", f"/dags/{dag_id}/dagRuns/{run['dag_run_id']}")
    if run["state"] == "failed":
        raise SystemExit(f"{dag_id} failed: check the Airflow UI")
    return (datetime.fromisoformat(run["end_date"]) - datetime.fromisoformat(run["start_date"])).total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cold", "warm"], required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--eval-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--eval-prompt", default="structured")
    ap.add_argument("--eval-limit", type=int, default=100)
    ap.add_argument("--out", default="results/timing_airflow.csv")
    a = ap.parse_args()
    bench_conf = {"models": [{"model": a.eval_model, "adapter": None}], "prompts": [a.eval_prompt],
                  "limit": a.eval_limit}
    for run in range(1, a.runs + 1):
        if a.mode == "cold":
            shutil.rmtree("data/interim", ignore_errors=True)
            shutil.rmtree(pathlib.Path(infer.PRED_ROOT) / f"{infer.model_name(a.eval_model)}__{a.eval_prompt}",
                          ignore_errors=True)
        etl = run_dag("preprocess_corpus", {"push": False})
        bench = run_dag("benchmark_models", bench_conf)
        secs = round(etl + bench, 1)
        print(f"{a.mode} run {run}: etl {etl:.1f}s + eval {bench:.1f}s = {secs}s")
        infer.append_csv(a.out, {"pipeline": f"airflow_{a.mode}", "run": run, "seconds": secs,
                                 "etl_seconds": round(etl, 1), "eval_seconds": round(bench, 1),
                                 "eval_model": a.eval_model, "eval_limit": a.eval_limit})


if __name__ == "__main__":
    main()
