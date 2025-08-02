from typing import List, Dict, Any
from core.vector_store import retrieve_top_k as semantic_retriever
from core.opensearch_store import search as keyword_retriever
from config import logger


def retrieve_hybrid(
    query: str, top_k_each: int = 20, final_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining BM25 (OpenSearch) and vector (Qdrant) results.
    Deduplicates by id and reranks by normalized score.
    """
    logger.info(f"🔍 Running hybrid search: '{query}'")

    vector_results = semantic_retriever(query, top_k=top_k_each)
    bm25_results = keyword_retriever(query, top_k=top_k_each)

    combined: Dict[str, Dict[str, Any]] = {}

    def normalize(scores):
        if not scores:
            return []
        max_score = max(scores)
        return [s / max_score if max_score > 0 else 0 for s in scores]

    vector_scores = normalize([r["score"] for r in vector_results])
    bm25_scores = normalize([r["score"] for r in bm25_results])

    for r, score in zip(vector_results, vector_scores):
        key = r["id"]
        combined[key] = {**r, "score_vector": score, "score_bm25": 0.0}

    for r, score in zip(bm25_results, bm25_scores):
        key = r["_id"]
        if key in combined:
            combined[key]["score_bm25"] = score
        else:
            combined[key] = {**r, "score_vector": 0.0, "score_bm25": score}

    for doc in combined.values():
        doc["hybrid_score"] = 0.7 * doc["score_vector"] + 0.3 * doc["score_bm25"]

    sorted_docs = sorted(
        combined.values(), key=lambda x: x["hybrid_score"], reverse=True
    )

    logger.info(f"✅ Hybrid search returned {len(sorted_docs)} unique results")
    return sorted_docs[:final_k]
