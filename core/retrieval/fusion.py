from __future__ import annotations
from typing import Dict, List, Iterable, Sequence
from core.retrieval.types import DocHit


def _merge_query_provenance(target: DocHit, source: DocHit) -> None:
    source_provenance = source.get("_query_provenance")
    if isinstance(source_provenance, list):
        existing = target.get("_query_provenance")
        if not isinstance(existing, list):
            existing = []
        for item in source_provenance:
            if isinstance(item, dict) and item not in existing:
                existing.append(item)
        target["_query_provenance"] = existing
    for key in ("_query_variant", "_query_text", "_query_channel"):
        if key not in target and key in source:
            target[key] = source[key]

def _normalize(scores: Iterable[float]) -> List[float]:
    scores = list(scores)
    if not scores:
        return []
    m = max(scores)
    return [s / m if m > 0 else 0.0 for s in scores]

def fuse_semantic_and_bm25(
    vector_results: Sequence[DocHit],
    bm25_results: Sequence[DocHit],
    w_vec: float = 0.7,
    w_bm25: float = 0.3,
) -> List[DocHit]:
    """
    Fuse by normalized scores; merge by id/_id and annotate source.
    Behavior matches your current implementation.
    """
    combined: Dict[str, DocHit] = {}

    v_norm = _normalize([r.get("score", 0.0) for r in vector_results])
    b_norm = _normalize([r.get("score", 0.0) for r in bm25_results])

    for r, s in zip(vector_results, v_norm):
        key = r.get("id")
        if not key:
            continue
        key_str = str(key)
        if key_str in combined:
            _merge_query_provenance(combined[key_str], r)
            if s >= combined[key_str].get("score_vector", 0.0):
                existing_bm25 = combined[key_str].get("score_bm25", 0.0)
                existing_source = combined[key_str].get("source", "semantic")
                combined[key_str] = {
                    **r,
                    "score_vector": s,
                    "score_bm25": existing_bm25,
                    "source": existing_source,
                }
                _merge_query_provenance(combined[key_str], r)
        else:
            combined[key_str] = {**r, "score_vector": s, "score_bm25": 0.0, "source": "semantic"}

    for r, s in zip(bm25_results, b_norm):
        key = r.get("_id")
        if not key:
            continue
        key_str = str(key)
        if key_str in combined:
            combined[key_str]["score_bm25"] = s
            combined[key_str]["source"] = "semantic/keyword"
            if "_bm25_variant_weight" in r:
                combined[key_str]["_bm25_variant_weight"] = r["_bm25_variant_weight"]
            if "_variant_rank" in r:
                combined[key_str]["_variant_rank"] = r["_variant_rank"]
            _merge_query_provenance(combined[key_str], r)
        else:
            combined[key_str] = {**r, "score_vector": 0.0, "score_bm25": s, "source": "keyword"}

    for doc in combined.values():
        doc["retrieval_score"] = w_vec * doc.get("score_vector", 0.0) + w_bm25 * doc.get("score_bm25", 0.0)

    return sorted(
        combined.values(),
        key=lambda x: (x.get("retrieval_score", 0.0), x.get("modified_at", "")),
        reverse=True,
    )

def dedup_by_checksum(sorted_docs: Sequence[DocHit]) -> List[DocHit]:
    """Exact behavior of your checksum dedup step."""
    unique_docs: List[DocHit] = []
    seen: set[object] = set()
    for d in sorted_docs:
        cs = d.get("checksum")
        if cs is None:
            unique_docs.append(d)
            continue
        if cs in seen:
            continue
        seen.add(cs)
        unique_docs.append(d)
    return unique_docs
