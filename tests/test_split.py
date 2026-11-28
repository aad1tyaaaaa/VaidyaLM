"""Tests for src/etl/split.py: python -m tests.test_split (needs pandas + pyarrow)."""
import pathlib
import tempfile

import pandas as pd

from src.etl import split


def _rows(n):
    return [{"id": str(i), "source": "t", "question": f"q{i}", "context": "", "response": f"r{i}", "label": None} for i in range(n)]


def test_split_sizes_and_no_overlap():
    train, dev = split.split(_rows(20), n_dev=5)
    assert len(dev) == 5 and len(train) == 15
    assert not {p["id"] for p in train} & {p["id"] for p in dev}


def test_split_is_deterministic_per_seed():
    a = split.split(_rows(30), n_dev=7, seed=1)
    b = split.split(_rows(30), n_dev=7, seed=1)
    c = split.split(_rows(30), n_dev=7, seed=2)
    assert a == b
    assert [p["id"] for p in a[1]] != [p["id"] for p in c[1]]


def test_split_dev_larger_than_data_leaves_empty_train():
    train, dev = split.split(_rows(3), n_dev=100)
    assert train == [] and len(dev) == 3


def test_write_roundtrip_columns():
    with tempfile.TemporaryDirectory() as d:
        train, dev = split.split(_rows(10), n_dev=2)
        split.write(train, dev, pathlib.Path(d) / "out")
        t = pd.read_parquet(pathlib.Path(d) / "out" / "train.parquet")
        assert list(t.columns) == split.COLUMNS and len(t) == 8


def test_dataset_card_lists_funnel_and_sources():
    card = split.dataset_card({"raw": 1200, "final": 900}, {"medquad": 500})
    assert "| raw | 1,200 |" in card and "| final | 900 |" in card and "| medquad | 500 |" in card


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("split tests ok")
