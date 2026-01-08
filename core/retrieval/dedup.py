from __future__ import annotations
from typing import Callable, List, Tuple, Sequence, Any
import numpy as np
from numpy.typing import NDArray
from core.retrieval.types import DocHit

FloatArray = NDArray[np.floating[Any]]

def _l2_norm(x: FloatArray) -> FloatArray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / n


def _cos(a: FloatArray, b: FloatArray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def collapse_near_duplicates(
    docs: Sequence[DocHit],
    embed_texts: Callable[[List[str]], Any],
    sim_threshold: float = 0.90,
    keep_limit: int = 64,
) -> Tuple[List[DocHit], List[DocHit]]:
    """
    Collapse near-duplicate passages by cosine similarity on embeddings.
    Returns (kept, duplicates). Keep one representative from each near-dup set.
    """
    if not docs:
        return [], []

    # Embed and L2-normalize
    embs = np.asarray(embed_texts([d.get("text", "") for d in docs]), dtype=float)
    embs = _l2_norm(embs)

    kept: List[DocHit] = []
    kept_e: List[FloatArray] = []
    dups: List[DocHit] = []

    for d, e in zip(docs, embs):
        is_dup = any(_cos(e, ke) >= sim_threshold for ke in kept_e)
        if is_dup:
            dups.append(d)
        else:
            kept.append(d)
            kept_e.append(e)
        if len(kept) >= keep_limit:
            # We still collect dups after reaching the keep limit (optional)
            continue

    return kept, dups
