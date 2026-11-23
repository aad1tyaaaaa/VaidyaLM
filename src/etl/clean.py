"""Normalize text and apply quality filters. Returns kept records + a Counter of drop reasons."""
import re
import unicodedata
from collections import Counter

LABELS = {"yes", "no", "maybe"}


def norm(t):
    t = unicodedata.normalize("NFKC", t or "")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines())
    return "\n".join(line for line in lines if line)


def is_english(t, min_ascii=0.9):
    # ponytail: ASCII-letter ratio as a language check; add a real detector (e.g. fasttext lid) if non-English sources appear
    letters = [c for c in t if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= min_ascii


def _jaccard(a, b):
    a, b = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(a & b) / len(a | b) if a | b else 1.0


def clean_record(r):
    q, a, ctx = norm(r["question"]), norm(r["answer"]), norm(r["context"])
    if a.lower().startswith(q.lower()):  # many MedQuAD answers start by repeating the question
        a = a[len(q):].strip()
    if not a:
        return None, "empty_answer"
    if not 10 <= len(q) <= 500:
        return None, "question_length"
    if not 20 <= len(a) <= 20000:
        return None, "answer_length"
    if r["source"] != "medquad" and len(ctx) < 100:
        return None, "context_length"
    if r["label"] is not None and r["label"] not in LABELS:
        return None, "bad_label"
    if not is_english(q + " " + a):
        return None, "language"
    if _jaccard(q, a) >= 0.8:
        return None, "answer_repeats_question"
    return {**r, "question": q, "answer": a, "context": ctx}, None


def clean_records(records):
    kept, reasons = [], Counter()
    for r in records:
        c, why = clean_record(r)
        if c:
            kept.append(c)
        else:
            reasons[why] += 1
    return kept, reasons
