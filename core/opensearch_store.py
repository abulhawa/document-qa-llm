from typing import List, cast
from core.retrieval.types import DocHit
from core.opensearch_client import get_client
from config import logger, CHUNKS_INDEX
from tracing import start_span, INPUT_VALUE, RETRIEVER, STATUS_OK


_KEYWORD_QUERY_FIELDS = [
    "text^1.0",
    "path^0.35",
    "filename^0.75",
    "filename.keyword^1.10",
]
_SIBLING_FETCH_FIELDS = [
    "text",
    "path",
    "filename",
    "checksum",
    "chunk_index",
    "modified_at",
    "page",
    "location_percent",
    "doc_type",
    "person_name",
    "authority_rank",
]

def search(query: str, top_k: int = 10) -> List[DocHit]:
    with start_span("Keyword retriever", kind=RETRIEVER) as span:
        logger.info(f"Searching OpenSearch for query: '{query}' with top_k={top_k}")
        span.set_attribute(INPUT_VALUE, query)
        span.set_attribute("top_k", top_k)

        client = get_client()
        response = client.search(
            index=CHUNKS_INDEX,
            body={
                "size": top_k * 3,  # fetch extra for dedup
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": _KEYWORD_QUERY_FIELDS,
                        "type": "best_fields",
                        "operator": "or",
                    }
                },
                "sort": [
                    {"_score": {"order": "desc"}},
                    {"modified_at": {"order": "desc"}},
                ],
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        logger.info(f"OpenSearch returned {len(hits)} hits before deduplication.")
        span.set_attribute("raw_hits", len(hits))

        results: List[DocHit] = []
        seen_checksums: set[object] = set()
        for hit in hits:
            src = hit.get("_source", {})
            checksum = src.get("checksum")
            if checksum in seen_checksums:
                continue
            seen_checksums.add(checksum)
            score = hit.get("_score")
            result = cast(
                DocHit,
                {
                    **src,
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                    "_id": hit.get("_id"),
                },
            )
            results.append(result)
            if len(results) >= top_k:
                break

        logger.info(f"Returning {len(results)} results after deduplication.")

        for i, doc in enumerate(results):
            doc_path = doc.get("path") or ""
            doc_score = doc.get("score")
            score_value = float(doc_score) if isinstance(doc_score, (int, float)) else 0.0
            span.set_attribute(f"retrieval.documents.{i}.document.id", doc_path)
            span.set_attribute(f"retrieval.documents.{i}.document.score", score_value)
            span.set_attribute(
                f"retrieval.documents.{i}.document.content", doc.get("text") or ""
            )
            span.set_attribute(
                f"retrieval.documents.{i}.document.metadata",
                [
                    f"Chunk index: {doc.get('chunk_index') or 'N/A'}",
                    f"Date modified: {doc.get('modified_at') or 'N/A'}",
                ],
            )
        span.set_status(STATUS_OK)
        return results


def fetch_sibling_chunks(doc: DocHit, limit: int = 12) -> List[DocHit]:
    checksum = str(doc.get("checksum") or "").strip()
    path = str(doc.get("path") or "").strip()
    if not checksum and not path:
        return []

    query = (
        {"term": {"checksum": {"value": checksum}}}
        if checksum
        else {"term": {"path.keyword": path}}
    )

    try:
        client = get_client()
        response = client.search(
            index=CHUNKS_INDEX,
            body={
                "size": max(1, int(limit)),
                "_source": _SIBLING_FETCH_FIELDS,
                "query": query,
                "sort": [
                    {"chunk_index": {"order": "asc", "missing": "_last"}},
                    {"_id": {"order": "asc"}},
                ],
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sibling chunk fetch failed for checksum=%s path=%s: %s",
            checksum or "N/A",
            path or "N/A",
            exc,
        )
        return []

    hits = response.get("hits", {}).get("hits", [])
    siblings: List[DocHit] = []
    for hit in hits:
        src = hit.get("_source", {})
        score = hit.get("_score")
        hit_id = hit.get("_id")
        siblings.append(
            cast(
                DocHit,
                {
                    **src,
                    "_id": hit_id,
                    "id": src.get("id") or hit_id,
                    "score": float(score) if isinstance(score, (int, float)) else 0.0,
                },
            )
        )
    return siblings
