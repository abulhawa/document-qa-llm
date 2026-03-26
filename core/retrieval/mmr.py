from __future__ import annotations
from typing import Any, Callable, List, Sequence
import numpy as np
from numpy.typing import NDArray
from core.retrieval.types import DocHit

FloatArray = NDArray[np.floating[Any]]

def _l2(x: FloatArray) -> FloatArray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / n


def _min_max(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if abs(hi - lo) < 1e-12:
        if hi > 0:
            return np.ones_like(arr)
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def mmr_select(
    query: str,
    docs: Sequence[DocHit],
    embed: Callable[[List[str]], Any],
    k: int = 8,
    lambda_mult: float = 0.6,
    retrieval_weight: float = 0.5,
) -> List[DocHit]:
    """
    Score-aware MMR over embeddings of doc['text'] with the query embedding.
    Blends query similarity with existing retrieval scores so novelty does not
    dominate low-priority documents. Falls back to first-k if embed fails.
    """
    if not docs:
        return []

    docs_list = list(docs)
    try:
        q = _l2(np.asarray(embed([query]), dtype=float))[0]
        D = _l2(np.asarray(embed([d.get("text", "") for d in docs_list]), dtype=float))
    except Exception:
        # If embedding API is busy/unavailable, gracefully degrade.
        return docs_list[:k]

    lambda_mult = min(max(lambda_mult, 0.0), 1.0)
    retrieval_weight = min(max(retrieval_weight, 0.0), 1.0)

    selected: List[int] = []
    candidates = list(range(len(docs_list)))

    # sim to query
    query_sims = ((D @ q) + 1.0) / 2.0
    retrieval_scores = _min_max(
        [
            float(doc.get("retrieval_score", doc.get("score", 0.0)) or 0.0)
            for doc in docs_list
        ]
    )
    relevance = (1.0 - retrieval_weight) * query_sims + retrieval_weight * retrieval_scores

    while candidates and len(selected) < k:
        best = None
        best_score = -1e9
        for i in candidates:
            div = max(float(D[i] @ D[j]) for j in selected) if selected else 0.0
            div = max(div, 0.0)
            score = lambda_mult * float(relevance[i]) - (1.0 - lambda_mult) * div
            if score > best_score:
                best_score = score
                best = i
        selected.append(best)           # type: ignore
        candidates.remove(best)         # type: ignore

    return [docs_list[i] for i in selected]
