"""Synthetic labels for unlabeled PubMed abstracts using a larger model (HF Inference).

Self-consistency: ask twice at different temperatures; keep only if both give the same parseable label
AND the reason is grounded in the abstract (word overlap). Off unless SYNTH_N > 0 and HF_TOKEN is set.
"""
import os
import re

from src.eval.prompts import structured_text
from src.eval.scorer import parse_label

MODEL = os.environ.get("SYNTH_MODEL", "Qwen/Qwen2.5-72B-Instruct")


def _words(t):
    return {w for w in re.findall(r"[a-z]{4,}", t.lower())}


def grounded(reason, context, min_overlap=0.5):
    # ponytail: word-overlap grounding check; swap for an NLI model if synthetic noise shows up in ablations
    w = _words(reason)
    return bool(w) and len(w & _words(context)) / len(w) >= min_overlap


def _reason(text):
    return text.split("\n", 1)[1].strip() if "\n" in text else ""


def label_rows(rows):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("synth: HF_TOKEN not set, skipping")
        return []
    from huggingface_hub import InferenceClient

    client = InferenceClient(MODEL, token=token)
    out = []
    for r in rows:
        msgs = [{"role": "user", "content": structured_text(r["question"], r["context"])}]
        try:
            a, b = (
                client.chat_completion(msgs, max_tokens=96, temperature=t).choices[0].message.content
                for t in (0.2, 0.8)
            )
        except Exception as e:  # rate limit / provider error: skip row, keep going
            print(f"synth: {r['id']} failed: {e}")
            continue
        label = parse_label(a)
        if label and label == parse_label(b) and grounded(_reason(a), r["context"]):
            out.append({"id": f"synth:{r['id']}", "source": "synthetic", "question": r["question"],
                        "context": r["context"], "answer": _reason(a), "label": label})
    return out
