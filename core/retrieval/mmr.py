from __future__ import annotations
from typing import List, Callable, Sequence
import numpy as np
from core.retrieval.types import DocHit

def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / n

def mmr_select(
    query: str,
    docs: Sequence[DocHit],
    embed: Callable[[List[str]], np.ndarray],
    k: int = 8,
    lambda_mult: float = 0.6,
) -> List[DocHit]:
    """
    MMR over embeddings of doc['text'] with the query embedding.
    Assumes 'text' exists. Falls back to first-k if embed fails.
    """
    if not docs:
        return []

    docs_list = list(docs)
    try:
        q = _l2(embed([query]))[0]
        D = _l2(embed([d.get("text", "") for d in docs_list]))
    except Exception:
        # If embedding API is busy/unavailable, gracefully degrade.
        return docs_list[:k]

    selected: List[int] = []
    candidates = list(range(len(docs_list)))

    # sim to query
    sims_q = (D @ q).tolist()

    while candidates and len(selected) < k:
        best = None
        best_score = -1e9
        for i in candidates:
            div = max(float(D[i] @ D[j]) for j in selected) if selected else 0.0
            score = lambda_mult * sims_q[i] - (1 - lambda_mult) * div
            if score > best_score:
                best_score = score
                best = i
        selected.append(best)           # type: ignore
        candidates.remove(best)         # type: ignore

    return [docs_list[i] for i in selected]
