"""Batched greedy inference with a prediction cache keyed by (model, prompt, example_id).

    python -m src.eval.infer --prompt structured
    python -m src.eval.infer --prompt few_shot --limit 50                 # quick smoke test
    python -m src.eval.infer --prompt structured --adapter user/repo      # fine-tuned model
    python -m src.eval.infer --prompt structured --dataset dev           # dev split built by the ETL

Cache layout: results/predictions/<model>__<prompt>/part-*.jsonl, one {"id","gold","output"} per line.
Reruns only generate ids that aren't cached yet.
"""
import argparse
import csv
import hashlib
import json
import pathlib

from src.eval import prompts, scorer

PRED_ROOT = pathlib.Path("results/predictions")


def load_examples(name="pqa_labeled", limit=None, dev_source="data/final"):
    from datasets import load_dataset

    if name == "pqa_labeled":
        rows = [prompts.from_hf(e) for e in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")]
    else:  # dev split from our ETL: local parquet dir or HF dataset repo
        src = pathlib.Path(dev_source)
        ds = (
            load_dataset("parquet", data_files=str(src / "dev.parquet"), split="train")
            if src.exists()
            else load_dataset(dev_source, split="dev")
        )
        rows = [{"id": r["id"], "question": r["question"], "context": r["context"], "label": r["label"]} for r in ds if r["label"]]
    return rows[:limit] if limit else rows


def model_name(model, adapter=None):
    return model.split("/")[-1] + (f"+{adapter.rstrip('/').split('/')[-1]}" if adapter else "")


def read_cache(key_dir):
    out = {}
    for part in sorted(pathlib.Path(key_dir).glob("part-*.jsonl")):
        for line in part.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def load_model(model_id, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16 if cuda else torch.float32, device_map="auto" if cuda else None
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.eval(), tok


def generate(model, tok, texts, batch_size=16, max_new_tokens=64):
    import torch

    torch.manual_seed(0)
    outs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"{len(outs)}/{len(texts)}", flush=True)
    return outs


def run(model_id, adapter, prompt, examples, batch_size=16, max_new_tokens=64, pred_root=PRED_ROOT):
    """Returns [{"id","gold","output"}] aligned with `examples`. pred_root=None disables the cache."""
    key_dir = pathlib.Path(pred_root) / f"{model_name(model_id, adapter)}__{prompt}" if pred_root else None
    cached = read_cache(key_dir) if key_dir else {}
    todo = [ex for ex in examples if ex["id"] not in cached]
    if todo:
        model, tok = load_model(model_id, adapter)
        shots = prompts.load_shots() if prompt == "few_shot" else ()
        build = prompts.TEMPLATES[prompt]
        texts = [
            tok.apply_chat_template([{"role": "user", "content": build(ex, shots)}], tokenize=False, add_generation_prompt=True)
            for ex in todo
        ]
        new = [
            {"id": ex["id"], "gold": ex["label"], "output": o}
            for ex, o in zip(todo, generate(model, tok, texts, batch_size, max_new_tokens))
        ]
        cached.update({r["id"]: r for r in new})
        if key_dir:
            key_dir.mkdir(parents=True, exist_ok=True)
            # part name = hash of its ids -> parallel shards never overwrite each other
            name = hashlib.sha1("".join(r["id"] for r in new).encode()).hexdigest()[:10]
            with open(key_dir / f"part-{name}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
    return [cached[ex["id"]] for ex in examples]


def append_csv(path, row):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", help="LoRA adapter path or HF repo (Phase 4)")
    ap.add_argument("--prompt", choices=prompts.TEMPLATES, required=True)
    ap.add_argument("--dataset", choices=["pqa_labeled", "dev"], default="pqa_labeled")
    ap.add_argument("--dev-source", default="data/final", help="local dir or HF dataset repo with the dev split")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="results/baseline.csv")
    a = ap.parse_args()

    examples = load_examples(a.dataset, a.limit, a.dev_source)
    rows = run(a.model, a.adapter, a.prompt, examples, a.batch_size, a.max_new_tokens)
    m = scorer.score([r["gold"] for r in rows], [r["output"] for r in rows])
    row = {"model": model_name(a.model, a.adapter), "prompt": a.prompt, "dataset": a.dataset, **m}
    print(row)
    append_csv(a.out, row)


if __name__ == "__main__":
    main()
