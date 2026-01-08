from typing import List, Dict, Any, cast
from core.retrieval.types import DocHit
from core.opensearch_client import get_client
from config import logger, CHUNKS_INDEX
from tracing import start_span, INPUT_VALUE, RETRIEVER, STATUS_OK


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
                "query": {"match": {"text": {"query": query, "operator": "or"}}},
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
        seen_checksums = set()
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
