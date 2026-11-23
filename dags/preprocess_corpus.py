"""ETL DAG: raw medical text -> validated instruction pairs on the HF Hub.

ingest_raw -> parse.expand(shards) -> clean.expand -> dedupe_minhash -> chunk
           -> build_instruction_pairs -> validate_schema -> split_and_push
"""
import os
from datetime import datetime

from airflow.decorators import dag, task
from airflow.models.param import Param


@dag(
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["vaidyalm"],
    params={"push": Param(True, type="boolean", description="push the dataset to HF_DATASET_REPO")},
)
def preprocess_corpus():
    @task
    def ingest_raw():
        from src.etl import ingest

        return ingest.run()

    @task(max_active_tis_per_dagrun=8)
    def parse_shard(shard):
        from src.etl import stages

        return stages.parse_stage(shard)

    @task(max_active_tis_per_dagrun=8)
    def clean_shard(parsed):
        from src.etl import stages

        return stages.clean_stage(parsed)

    @task
    def dedupe_minhash(cleaned):
        from src.etl import stages

        return stages.dedupe_stage(list(cleaned))

    @task
    def chunk(d):
        from src.etl import stages

        return stages.chunk_stage(d)

    @task
    def build_instruction_pairs(d):
        from src.etl import stages

        return stages.build_pairs_stage(d)

    @task
    def validate_schema(d):
        from src.etl import stages

        return stages.validate_stage(d)

    @task
    def split_and_push(d, params=None):
        from src.etl import stages

        repo = os.environ.get("HF_DATASET_REPO") if params["push"] and os.environ.get("HF_TOKEN") else None
        funnel = stages.split_and_push_stage(d, repo)
        print("data funnel:", funnel)
        return funnel

    parsed = parse_shard.expand(shard=ingest_raw())
    cleaned = clean_shard.expand(parsed=parsed)
    split_and_push(validate_schema(build_instruction_pairs(chunk(dedupe_minhash(cleaned)))))


preprocess_corpus()
