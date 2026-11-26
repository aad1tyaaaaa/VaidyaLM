"""Tests for src/eval/report.py: python -m tests.test_report (needs scikit-learn, pandas)."""
import json
import pathlib
import tempfile

from src.eval import report


def _write(pred_root, key, rows, part="part-0000.jsonl"):
    d = pathlib.Path(pred_root) / key
    d.mkdir(parents=True, exist_ok=True)
    (d / part).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_build():
    with tempfile.TemporaryDirectory() as tmp:
        pred, out = pathlib.Path(tmp) / "pred", pathlib.Path(tmp) / "out"
        _write(pred, "base__structured", [
            {"id": "1", "gold": "yes", "output": "Answer: yes\nReason: x"},
            {"id": "2", "gold": "no", "output": "Answer: yes\nReason: wrong"},
            {"id": "3", "gold": "maybe", "output": "I cannot say"},          # unparseable -> "invalid"
        ])
        _write(pred, "empty__zero_shot", [])                                   # no rows -> skipped
        rows = report.build(pred_root=pred, out_dir=out)

        assert [(r["model"], r["prompt"], r["n"]) for r in rows] == [("base", "structured", 3)], rows
        assert abs(rows[0]["accuracy"] - 1 / 3) < 1e-9 and abs(rows[0]["parse_fail_pct"] - 100 / 3) < 1e-9, rows
        md = (out / "benchmark.md").read_text(encoding="utf-8")
        assert "| base | structured | 3 | 0.333 |" in md, md
        assert "| yes | 1 | 0 | 0 | 0 |" in md and "| no | 1 | 0 | 0 | 0 |" in md, md  # cols: yes no maybe invalid
        assert "| maybe | 0 | 0 | 0 | 1 |" in md, md
        assert (out / "benchmark.csv").exists()


def test_ids_filter():
    with tempfile.TemporaryDirectory() as tmp:
        pred, out = pathlib.Path(tmp) / "pred", pathlib.Path(tmp) / "out"
        _write(pred, "m__zero_shot", [
            {"id": "1", "gold": "yes", "output": "yes"},
            {"id": "2", "gold": "no", "output": "yes"},
        ])
        rows = report.build(ids=["1"], pred_root=pred, out_dir=out)
        assert rows[0]["n"] == 1 and rows[0]["accuracy"] == 1.0, rows


if __name__ == "__main__":
    test_build()
    test_ids_filter()
    print("report ok")
