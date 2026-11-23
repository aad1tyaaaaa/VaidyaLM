"""Score cached predictions -> results/benchmark.md (comparison table + confusion matrices) and benchmark.csv.

    python -m src.eval.report                      # every model__prompt in results/predictions
"""
import pathlib

import pandas as pd
from sklearn.metrics import confusion_matrix

from src.eval import infer, scorer

COLS = scorer.LABELS + ["invalid"]


def build(keys=None, ids=None, pred_root=infer.PRED_ROOT, out_dir="results"):
    pred_root = pathlib.Path(pred_root)
    keys = keys or sorted(p.name for p in pred_root.iterdir() if p.is_dir())
    rows, cms = [], []
    for key in keys:
        cached = infer.read_cache(pred_root / key)
        sel = [cached[i] for i in ids if i in cached] if ids else list(cached.values())
        if not sel:
            continue
        golds, outs = [r["gold"] for r in sel], [r["output"] for r in sel]
        model, prompt = key.split("__")
        rows.append({"model": model, "prompt": prompt, **scorer.score(golds, outs)})
        preds = [scorer.parse_label(o) or "invalid" for o in outs]
        cms.append((key, confusion_matrix(golds, preds, labels=COLS)[:3]))  # gold is never "invalid"

    df = pd.DataFrame(rows)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False)

    md = ["# Benchmark: PubMedQA `pqa_labeled`", "", "| Model | Prompt | n | Accuracy | Macro-F1 | Parse fail % |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['prompt']} | {r['n']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['parse_fail_pct']:.1f} |"
           for r in rows]
    md += ["", "## Confusion matrices (rows = gold, columns = predicted)"]
    for key, cm in cms:
        md += ["", f"**{key}**", "", "| gold \\ pred | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
        md += [f"| {g} | " + " | ".join(str(v) for v in cm[i]) + " |" for i, g in enumerate(scorer.LABELS)]
    (out / "benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    for r in build():
        print(r)
