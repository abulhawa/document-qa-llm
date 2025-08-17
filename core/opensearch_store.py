from typing import List, Dict, Any
from core.opensearch_client import get_client
from config import (
    logger,
    OPENSEARCH_INDEX,
    OS_MIN_CHILDREN,
    OS_INNER_HITS,
)
from tracing import start_span, INPUT_VALUE, RETRIEVER, STATUS_OK


def search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    with start_span("Keyword retriever", kind=RETRIEVER) as span:
        logger.info(f"Searching OpenSearch for query: '{query}' with top_k={top_k}")
        span.set_attribute(INPUT_VALUE, query)
        span.set_attribute("top_k", top_k)

        client = get_client()
        body = {
            "size": top_k * 3,
            "query": {
                "has_child": {
                    "type": "chunk",
                    "score_mode": "sum",
                    "min_children": OS_MIN_CHILDREN,
                    "query": {
                        "simple_query_string": {
                            "query": query,
                            "default_operator": "and",
                        }
                    },
                    "inner_hits": {
                        "size": OS_INNER_HITS,
                        "sort": [{"_score": {"order": "desc"}}],
                        "highlight": {
                            "fields": {"text": {"pre_tags": ["<mark>"], "post_tags": ["</mark>"]}}
                        },
                    },
                }
            },
        }
        response = client.search(index=OPENSEARCH_INDEX, body=body)
        hits = response.get("hits", {}).get("hits", [])
        logger.info(f"OpenSearch returned {len(hits)} parent hits before deduplication.")
        span.set_attribute("raw_hits", len(hits))

        results = []
        seen_checksums = set()
        for hit in hits:
            src = hit.get("_source", {})
            checksum = src.get("checksum")
            if checksum in seen_checksums:
                continue
            seen_checksums.add(checksum)

            inner = (
                hit.get("inner_hits", {})
                .get("chunk", {})
                .get("hits", {})
                .get("hits", [])
            )
            snippets = []
            for c in inner:
                csrc = c.get("_source", {})
                snippets.append(
                    {
                        "text": csrc.get("text", ""),
                        "page": csrc.get("page"),
                        "chunk_index": csrc.get("chunk_index"),
                        "highlight": c.get("highlight", {}).get("text", [csrc.get("text", "")])[0],
                        "score": c.get("_score"),
                    }
                )

            results.append({**src, "score": hit.get("_score"), "chunks": snippets, "doc_id": src.get("doc_id")})
            if len(results) >= top_k:
                break

        logger.info(f"Returning {len(results)} parent results after deduplication.")

        for i, doc in enumerate(results):
            span.set_attribute(f"retrieval.documents.{i}.document.id", doc.get("path"))
            span.set_attribute(f"retrieval.documents.{i}.document.score", doc["score"])
        span.set_status(STATUS_OK)
        return results
