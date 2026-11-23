"""Record -> instruction pair {id, source, question, context, response, label}.

Labeled rows get the exact structured-output format the eval prompt asks for, so training teaches the format.
"""
import re


def first_sentence(t):
    return re.split(r"(?<=[.!?])\s", t.strip(), maxsplit=1)[0]


def to_pair(r):
    response = f"Answer: {r['label']}\nReason: {first_sentence(r['answer'])}" if r["label"] else r["answer"]
    return {k: r[k] for k in ("id", "source", "question", "context", "label")} | {"response": response}
