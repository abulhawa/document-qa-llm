from typing import List, Dict, Any

from core.vector_store import retrieve_top_k as semantic_retriever
from core.opensearch_store import search as keyword_retriever
from config import logger, HYBRID_W_OS, HYBRID_W_VEC
from tracing import start_span, STATUS_OK, RETRIEVER, INPUT_VALUE


def retrieve_hybrid(
    query: str, top_k_each: int = 20, final_k: int = 5
) -> List[Dict[str, Any]]:
    """Hybrid search combining OpenSearch (BM25) and Qdrant (semantic).

    Returns document-level results with fused scores and evidence snippets.
    """

    if not query or not query.strip():
        return []

    logger.info(f"🔍 Running hybrid search: '{query}'")

    with start_span("Hybrid retrieval", kind=RETRIEVER) as span:
        span.set_attribute(INPUT_VALUE, query)

        vector_results = semantic_retriever(query, top_k=top_k_each)
        bm25_results = keyword_retriever(query, top_k=top_k_each)

        # Group semantic chunks by doc_id and keep highest scoring snippet
        semantic_by_doc: Dict[str, Dict[str, Any]] = {}
        for r in vector_results:
            doc_id = r.get("doc_id") or r.get("path")
            prev = semantic_by_doc.get(doc_id)
            if not prev or r["score"] > prev["score"]:
                semantic_by_doc[doc_id] = r

        combined: Dict[str, Dict[str, Any]] = {}

        for r in bm25_results:
            doc_id = r.get("doc_id") or r.get("path")
            combined[doc_id] = {
                **r,
                "bm25": r.get("score", 0.0),
                "vec": 0.0,
                "semantic_snippet": None,
            }

        for doc_id, r in semantic_by_doc.items():
            if doc_id in combined:
                combined[doc_id]["vec"] = r.get("score", 0.0)
                combined[doc_id]["semantic_snippet"] = r.get("text")
            else:
                combined[doc_id] = {
                    **r,
                    "bm25": 0.0,
                    "vec": r.get("score", 0.0),
                    "chunks": [],
                    "semantic_snippet": r.get("text"),
                }

        for doc in combined.values():
            doc["final_score"] = HYBRID_W_OS * doc.get("bm25", 0.0) + HYBRID_W_VEC * doc.get(
                "vec", 0.0
            )
            evidence = list(doc.get("chunks", []))
            if doc.get("semantic_snippet"):
                evidence.append(
                    {
                        "text": doc["semantic_snippet"],
                        "score": doc.get("vec", 0.0),
                        "source": "semantic",
                    }
                )
            doc["evidence"] = evidence

        sorted_docs = sorted(
            combined.values(),
            key=lambda x: (x["final_score"], x.get("modified_at", "")),
            reverse=True,
        )

        unique_docs: List[Dict[str, Any]] = []
        seen_checksums = set()
        for doc in sorted_docs:
            cs = doc.get("checksum")
            if cs and cs in seen_checksums:
                continue
            if cs:
                seen_checksums.add(cs)
            unique_docs.append(doc)

        for i, doc in enumerate(unique_docs):
            span.set_attribute(f"retrieval.documents.{i}.document.id", doc.get("path"))
            span.set_attribute(
                f"retrieval.documents.{i}.document.score", doc["final_score"]
            )

        span.set_status(STATUS_OK)

        logger.info(
            f"✅ Hybrid search returned {len(unique_docs)} results after checksum dedup"
        )

    return unique_docs[:final_k]

