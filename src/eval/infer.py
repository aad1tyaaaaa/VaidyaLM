"""Batched greedy inference on PubMedQA pqa_labeled; writes predictions + appends metrics to a results CSV.

    python -m src.eval.infer --prompt structured
    python -m src.eval.infer --prompt few_shot --limit 50      # quick smoke test
"""
import argparse
import csv
import json
import pathlib

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval import prompts, scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter path/repo (Phase 4)")
    ap.add_argument("--prompt", choices=prompts.TEMPLATES, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="results/baseline.csv")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="auto")
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    if a.limit:
        ds = ds.select(range(a.limit))
    shots = prompts.load_shots() if a.prompt == "few_shot" else ()
    build = prompts.TEMPLATES[a.prompt]
    texts = [
        tok.apply_chat_template([{"role": "user", "content": build(ex, shots)}], tokenize=False, add_generation_prompt=True)
        for ex in ds
    ]

    outputs = []
    for i in range(0, len(texts), a.batch_size):
        enc = tok(texts[i : i + a.batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outputs += tok.batch_decode(gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        print(f"{len(outputs)}/{len(texts)}", flush=True)

    name = f"{a.model.split('/')[-1]}{'+' + a.adapter.split('/')[-1] if a.adapter else ''}"
    pred_dir = pathlib.Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / f"{name}__{a.prompt}.jsonl", "w", encoding="utf-8") as f:
        for ex, out in zip(ds, outputs):
            f.write(json.dumps({"pubid": ex["pubid"], "gold": ex["final_decision"], "output": out}) + "\n")

    m = scorer.score(ds["final_decision"], outputs)
    row = {"model": name, "prompt": a.prompt, **m}
    print(row)
    out = pathlib.Path(a.out)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row)
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
