"""Fit records into the training context window (max_seq_len 1024 tokens ~ 4k chars total)."""
MAX_ANSWER_CHARS = 1500
MAX_CONTEXT_CHARS = 2500


def truncate(text, max_chars):
    """Cut at the last paragraph/sentence boundary before max_chars (word boundary as fallback)."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    i = max(cut.rfind("\n"), cut.rfind(". "))
    return cut[: i + 1].strip() if i > max_chars // 2 else cut.rsplit(" ", 1)[0]


def chunk_record(r):
    return {**r, "answer": truncate(r["answer"], MAX_ANSWER_CHARS), "context": truncate(r["context"], MAX_CONTEXT_CHARS)}
