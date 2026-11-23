"""MinHash LSH near-duplicate removal + test-set leakage filter (one pass)."""
import re

from datasketch import MinHash, MinHashLSH

NUM_PERM = 64


def _key_text(r):
    # PubMedQA-style rows are identified by question+context; MedQuAD rows by question+answer
    return f"{r['question']} {r['context'] or r.get('answer', '')}"


def minhash(text):
    words = re.findall(r"\w+", text.lower())
    shingles = {" ".join(words[i : i + 3]) for i in range(max(1, len(words) - 2))}
    m = MinHash(num_perm=NUM_PERM)
    m.update_batch([s.encode() for s in shingles])
    return m


def dedupe(records, test_rows=(), threshold=0.85):
    """Returns (kept, n_duplicates, n_leaked). test_rows: [{question, context}] from pqa_labeled."""
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    for i, t in enumerate(test_rows):
        lsh.insert(f"test:{i}", minhash(_key_text(t)))
    kept, dup, leak = [], 0, 0
    for r in records:
        m = minhash(_key_text(r))
        hits = lsh.query(m)
        if any(h.startswith("test:") for h in hits):
            leak += 1
        elif hits:
            dup += 1
        else:
            lsh.insert(r["id"], m)
            kept.append(r)
    return kept, dup, leak
