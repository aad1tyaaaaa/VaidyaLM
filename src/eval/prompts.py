"""Prompt templates for PubMedQA. Each template takes one example and returns the user message text."""

STRUCTURED_RULE = (
    "Answer using only the context. Output `Answer: yes|no|maybe` on the first line, "
    "then a one-line reason."
)


def _ctx(ex):
    return "\n".join(ex["context"]["contexts"])


def zero_shot(ex, shots=()):
    return f"Context:\n{_ctx(ex)}\n\nQuestion: {ex['question']}\nAnswer with yes, no, or maybe."


def few_shot(ex, shots=()):
    demos = "\n\n".join(
        f"Context:\n{_ctx(s)}\n\nQuestion: {s['question']}\nAnswer: {s['final_decision']}" for s in shots
    )
    return f"{demos}\n\nContext:\n{_ctx(ex)}\n\nQuestion: {ex['question']}\nAnswer:"


def structured(ex, shots=()):
    return f"{STRUCTURED_RULE}\n\nContext:\n{_ctx(ex)}\n\nQuestion: {ex['question']}"


TEMPLATES = {"zero_shot": zero_shot, "few_shot": few_shot, "structured": structured}


def load_shots():
    """3 fixed demos from pqa_artificial (never the test set): yes, no, yes.
    pqa_artificial has no 'maybe' labels; swap in dev-set shots once Phase 3 builds it."""
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train[:200]")
    yes = [e for e in ds if e["final_decision"] == "yes"]
    no = [e for e in ds if e["final_decision"] == "no"]
    return [yes[0], no[0], yes[1]]
