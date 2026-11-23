"""Prompt templates for PubMedQA.

Every example is a flat dict: {"id", "question", "context", "label"} (context is a plain string).
Each template returns the user-message text.
"""

STRUCTURED_RULE = (
    "Answer using only the context. Output `Answer: yes|no|maybe` on the first line, "
    "then a one-line reason."
)


def structured_text(question, context):
    return f"{STRUCTURED_RULE}\n\nContext:\n{context}\n\nQuestion: {question}"


def zero_shot(ex, shots=()):
    return f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}\nAnswer with yes, no, or maybe."


def few_shot(ex, shots=()):
    demos = "\n\n".join(
        f"Context:\n{s['context']}\n\nQuestion: {s['question']}\nAnswer: {s['label']}" for s in shots
    )
    return f"{demos}\n\nContext:\n{ex['context']}\n\nQuestion: {ex['question']}\nAnswer:"


def structured(ex, shots=()):
    return structured_text(ex["question"], ex["context"])


TEMPLATES = {"zero_shot": zero_shot, "few_shot": few_shot, "structured": structured}


def format_user(row):
    """Training-time user message: structured prompt when there's a context (PubMedQA-style), else the bare question."""
    return structured_text(row["question"], row["context"]) if row["context"] else row["question"]


def from_hf(e):
    """PubMedQA HF row -> flat example."""
    return {
        "id": str(e["pubid"]),
        "question": e["question"],
        "context": "\n".join(e["context"]["contexts"]),
        "label": e["final_decision"],
    }


def load_shots():
    """3 fixed demos from pqa_artificial (never the test set): yes, no, yes.
    pqa_artificial has no 'maybe' labels."""
    from datasets import load_dataset

    ds = [from_hf(e) for e in load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train[:200]")]
    yes = [e for e in ds if e["label"] == "yes"]
    no = [e for e in ds if e["label"] == "no"]
    return [yes[0], no[0], yes[1]]
