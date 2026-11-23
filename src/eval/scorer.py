"""Parse yes/no/maybe from model output and compute accuracy, macro-F1, parse-failure rate."""
import re

from sklearn.metrics import accuracy_score, f1_score

LABELS = ["yes", "no", "maybe"]
_ANSWER = re.compile(r"answer\W*(yes|no|maybe)\b", re.I)
_BARE = re.compile(r"\b(yes|no|maybe)\b", re.I)


def parse_label(text):
    # ponytail: falls back to the first bare yes/no/maybe; can misread rambling answers. Parse-fail % exposes it.
    m = _ANSWER.search(text) or _BARE.search(text)
    return m.group(1).lower() if m else None


def score(golds, outputs):
    preds = [parse_label(o) for o in outputs]
    filled = [p or "invalid" for p in preds]  # unparseable counts as wrong
    return {
        "n": len(golds),
        "accuracy": accuracy_score(golds, filled),
        "macro_f1": f1_score(golds, filled, labels=LABELS, average="macro", zero_division=0),
        "parse_fail_pct": 100 * preds.count(None) / len(preds),
    }


if __name__ == "__main__":
    assert parse_label("Answer: Yes\nBecause...") == "yes"
    assert parse_label("**Answer:** maybe") == "maybe"
    assert parse_label("No, the study shows...") == "no"
    assert parse_label("I cannot tell.") is None
    assert parse_label("Answer: no. Not yes.") == "no"
    m = score(["yes", "no", "maybe", "yes"], ["Answer: yes", "Answer: no", "unsure", "Answer: no"])
    assert m["accuracy"] == 0.5 and m["parse_fail_pct"] == 25.0, m
    print("scorer ok", m)
