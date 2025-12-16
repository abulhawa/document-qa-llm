from __future__ import annotations
from typing import List, Dict, Any, Callable
import numpy as np

def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / n

def mmr_select(
    query: str,
    docs: List[Dict[str, Any]],
    embed: Callable[[List[str]], np.ndarray],
    k: int = 8,
    lambda_mult: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    MMR over embeddings of doc['text'] with the query embedding.
    Assumes 'text' exists. Falls back to first-k if embed fails.
    """
    if not docs:
        return []

    try:
        q = _l2(embed([query]))[0]
        D = _l2(embed([d.get("text", "") for d in docs]))
    except Exception:
        # If embedding API is busy/unavailable, gracefully degrade.
        return docs[:k]

    selected: List[int] = []
    candidates = list(range(len(docs)))

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

    return [docs[i] for i in selected]
