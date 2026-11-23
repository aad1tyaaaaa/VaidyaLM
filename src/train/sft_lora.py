"""QLoRA instruction tuning driven by a YAML config.

    python -m src.train.sft_lora src/train/configs/r16.yaml --dataset you/vaidyalm-instruct --hub-user you

Loss is computed on the assistant response only (prompt tokens are masked with -100).
The user message is built by src.eval.prompts.format_user -> identical format to the structured eval prompt.
"""
import argparse
import json
import os
import pathlib

import yaml

from src.eval.prompts import format_user

DEFAULTS = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dataset": "data/final",
    "sources": ["medquad", "pubmedqa_artificial", "synthetic"],
    "max_train_rows": 10000,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lr": 2e-4,
    "epochs": 2,
    "batch_size": 4,
    "grad_accum": 4,
    "max_seq_len": 1024,
    "save_steps": 100,
    "seed": 0,
    "report_to": "none",
}


def load_config(path, **overrides):
    cfg = DEFAULTS | yaml.safe_load(pathlib.Path(path).read_text()) | {k: v for k, v in overrides.items() if v is not None}
    cfg.setdefault("run_name", pathlib.Path(path).stem)
    return cfg


def load_splits(dataset):
    from datasets import load_dataset

    p = pathlib.Path(dataset)
    if p.exists():
        return load_dataset("parquet", data_files={"train": str(p / "train.parquet"), "dev": str(p / "dev.parquet")})
    return load_dataset(dataset)


def encode(row, tok, max_len):
    """Tokenize prompt and response separately so the label mask lines up exactly."""
    msgs = [{"role": "user", "content": format_user(row)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": row["response"]}], tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dataset", help="local dir with train/dev parquet, or HF dataset repo")
    ap.add_argument("--hub-user", help="push adapter to <hub-user>/vaidyalm-qwen1.5b-lora-<run_name>")
    ap.add_argument("--output-root", default="outputs")
    ap.add_argument("--max-train-rows", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config, dataset=a.dataset, max_train_rows=a.max_train_rows)
    print(json.dumps(cfg, indent=1))

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    ds = load_splits(cfg["dataset"])
    train = ds["train"].filter(lambda r: r["source"] in cfg["sources"]).shuffle(seed=cfg["seed"])
    if cfg["max_train_rows"]:
        train = train.select(range(min(cfg["max_train_rows"], len(train))))
    cols = train.column_names
    train = train.map(encode, fn_kwargs={"tok": tok, "max_len": cfg["max_seq_len"]}, remove_columns=cols)
    dev = ds["dev"].map(encode, fn_kwargs={"tok": tok, "max_len": cfg["max_seq_len"]}, remove_columns=cols)
    # drop rows where truncation cut off the whole response (nothing to learn from)
    train, dev = (d.filter(lambda r: any(x != -100 for x in r["labels"])) for d in (train, dev))
    print(f"train rows: {len(train)}, dev rows: {len(dev)}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    out = pathlib.Path(a.output_root) / cfg["run_name"]
    hub_repo = f"{a.hub_user}/vaidyalm-qwen1.5b-lora-{cfg['run_name']}" if a.hub_user else None
    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=cfg["epochs"],
        fp16=True,  # T4 has no bf16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg["save_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        seed=cfg["seed"],
        report_to=cfg["report_to"],
        run_name=cfg["run_name"],
        # checkpoints also go to the Hub so a dropped Kaggle session can resume
        push_to_hub=bool(hub_repo and os.environ.get("HF_TOKEN")),
        hub_model_id=hub_repo,
        hub_strategy="checkpoint",
        hub_private_repo=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))

    ckpt = get_last_checkpoint(out) if out.exists() else None
    if not ckpt and args.push_to_hub:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(hub_repo, allow_patterns=["last-checkpoint/*"], local_dir=out)
            ckpt = str(out / "last-checkpoint") if (out / "last-checkpoint").exists() else None
        except Exception:
            pass  # fresh run: repo doesn't exist yet
    if ckpt:
        print("resuming from", ckpt)
    trainer.train(resume_from_checkpoint=ckpt)

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    logs = pathlib.Path("results/train_logs")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{cfg['run_name']}.json").write_text(json.dumps({"config": cfg, "log_history": trainer.state.log_history}, indent=1))
    if args.push_to_hub:
        trainer.push_to_hub()
        print("adapter pushed to", hub_repo)


if __name__ == "__main__":
    main()
