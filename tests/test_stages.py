"""Checks the ETL cache logic (content hash + sidecar file). Run: python -m tests.test_stages"""
import pathlib
import tempfile

import pandas as pd

from src.etl import parse, stages


def _rec(i, answer=None):
    return {c: None for c in parse.COLUMNS} | {"id": f"r:{i}", "answer": answer}


def demo():
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)

        # _hash: same bytes -> same hash; any change in content or file name -> different hash
        (d / "shard").mkdir()
        (d / "shard" / "a.xml").write_text("one")
        h1 = stages._hash(d / "shard")
        assert h1 == stages._hash(d / "shard")
        (d / "shard" / "a.xml").write_text("two")
        h2 = stages._hash(d / "shard")
        assert h1 != h2, "content change not detected"
        (d / "shard" / "a.xml").rename(d / "shard" / "b.xml")
        assert stages._hash(d / "shard") != h2, "rename not detected"

        # _cached: first call computes, second with same key skips, a new key recomputes
        out = d / "out" / "x.parquet"
        calls = []

        def compute():
            calls.append(1)
            return [_rec(1, "a"), _rec(2)]

        assert stages._cached(out, "k1", compute) == (2, False)
        assert stages._cached(out, "k1", compute) == (2, True) and len(calls) == 1
        assert stages._cached(out, "k2", compute) == (2, False) and len(calls) == 2
        # a missing sidecar means we can't trust the parquet -> recompute
        out.with_suffix(".hash").unlink()
        assert stages._cached(out, "k2", compute)[1] is False and len(calls) == 3

        # _read: missing strings come back as None, not NaN (NaN is truthy and breaks `if r["label"]`)
        recs = stages._read(out)
        assert recs[0]["answer"] == "a" and recs[1]["answer"] is None, recs
        assert not recs[1]["answer"]
    print("stages ok")


if __name__ == "__main__":
    demo()
