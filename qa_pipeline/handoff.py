from __future__ import annotations

from typing import Optional, Sequence

from qa_pipeline.types import RetrievedDocument

DEFAULT_DYNAMIC_TOKEN_BUDGET = 1200
DEFAULT_DYNAMIC_MIN_CHUNKS = 3


def estimate_tokens(text: str) -> int:
    normalized = str(text or "")
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def pack_docs_by_token_budget(
    docs: Sequence[RetrievedDocument],
    *,
    token_budget: int,
    min_chunks: int = DEFAULT_DYNAMIC_MIN_CHUNKS,
    max_chunks: Optional[int] = None,
) -> tuple[list[RetrievedDocument], int]:
    if token_budget <= 0:
        return list(docs), sum(estimate_tokens(doc.text) for doc in docs)

    packed: list[RetrievedDocument] = []
    used_tokens = 0
    min_chunks = max(0, int(min_chunks))

    for doc in docs:
        if max_chunks is not None and len(packed) >= max_chunks:
            break
        doc_tokens = estimate_tokens(doc.text)
        if len(packed) < min_chunks:
            packed.append(doc)
            used_tokens += doc_tokens
            continue
        if used_tokens + doc_tokens > token_budget:
            continue
        packed.append(doc)
        used_tokens += doc_tokens

    if not packed and docs:
        first = docs[0]
        packed = [first]
        used_tokens = estimate_tokens(first.text)

    return packed, used_tokens
