from typing import List, Dict, Any
from core.opensearch_client import get_client
from config import logger, OPENSEARCH_INDEX
from tracing import start_span, INPUT_VALUE, RETRIEVER, STATUS_OK


def search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    with start_span("Keyword retriever", kind=RETRIEVER) as span:
        logger.info(f"Searching OpenSearch for query: '{query}' with top_k={top_k}")
        span.set_attribute(INPUT_VALUE, query)
        span.set_attribute("top_k", top_k)

        client = get_client()
        response = client.search(
            index=OPENSEARCH_INDEX,
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

        results = []
        seen_checksums = set()
        for hit in hits:
            src = hit.get("_source", {})
            checksum = src.get("checksum")
            if checksum in seen_checksums:
                continue
            seen_checksums.add(checksum)
            results.append({**src, "score": hit.get("_score"), "_id": hit.get("_id")})
            if len(results) >= top_k:
                break

        logger.info(f"Returning {len(results)} results after deduplication.")

        for i, doc in enumerate(results):
            span.set_attribute(f"retrieval.documents.{i}.document.id", doc["path"])
            span.set_attribute(f"retrieval.documents.{i}.document.score", doc["score"])
            span.set_attribute(f"retrieval.documents.{i}.document.content", doc["text"])
            span.set_attribute(
                f"retrieval.documents.{i}.document.metadata",
                [
                    f"Chunk index: {doc['chunk_index']}",
                    f"Date modified: {doc['modified_at']}",
                ],
            )
        span.set_status(STATUS_OK)
        return results
