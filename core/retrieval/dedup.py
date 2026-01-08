from __future__ import annotations
from typing import Callable, List, Tuple, Sequence
import numpy as np
from core.retrieval.types import DocHit


def _l2_norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def collapse_near_duplicates(
    docs: Sequence[DocHit],
    embed_texts: Callable[[List[str]], np.ndarray],
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
    embs = embed_texts([d.get("text", "") for d in docs])
    embs = _l2_norm(embs)

    kept: List[DocHit] = []
    kept_e: List[np.ndarray] = []
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
