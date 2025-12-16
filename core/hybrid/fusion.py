from __future__ import annotations
from typing import Dict, List, Any, Iterable
from config import logger

def _normalize(scores: Iterable[float]) -> List[float]:
    scores = list(scores)
    if not scores:
        return []
    m = max(scores)
    return [s / m if m > 0 else 0.0 for s in scores]

def fuse_semantic_and_bm25(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    w_vec: float = 0.7,
    w_bm25: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Fuse by normalized scores; merge by id/_id and annotate source.
    Behavior matches your current implementation.
    """
    combined: Dict[str, Dict[str, Any]] = {}

    v_norm = _normalize([r["score"] for r in vector_results])
    b_norm = _normalize([r["score"] for r in bm25_results])

    for r, s in zip(vector_results, v_norm):
        key = r["id"]
        combined[key] = {**r, "score_vector": s, "score_bm25": 0.0, "source": "semantic"}

    for r, s in zip(bm25_results, b_norm):
        key = r["_id"]
        if key in combined:
            combined[key]["score_bm25"] = s
            combined[key]["source"] = "semantic/keyword"
        else:
            combined[key] = {**r, "score_vector": 0.0, "score_bm25": s, "source": "keyword"}

    for doc in combined.values():
        doc["hybrid_score"] = w_vec * doc["score_vector"] + w_bm25 * doc["score_bm25"]

    return sorted(combined.values(), key=lambda x: (x["hybrid_score"], x.get("modified_at", "")), reverse=True)

def dedup_by_checksum(sorted_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exact behavior of your checksum dedup step."""
    unique_docs: List[Dict[str, Any]] = []
    seen = set()
    for d in sorted_docs:
        cs = d.get("checksum")
        if cs in seen:
            continue
        seen.add(cs)
        unique_docs.append(d)
    return unique_docs
