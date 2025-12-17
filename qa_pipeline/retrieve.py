from typing import List

from config import logger
from core.hybrid.pipeline import retrieve_hybrid
from qa_pipeline.types import RetrievalResult, RetrievedDocument


def retrieve_context(query: str, top_k: int) -> RetrievalResult:
    logger.info("🔍 Running semantic search for user question...")
    raw_results = retrieve_hybrid(query, top_k_each=20, final_k=top_k)

    documents: List[RetrievedDocument] = []
    for result in raw_results:
        if "clarify" in result:
            # Pass through clarification requests from hybrid variants generator
            logger.info("Retriever requested clarification: %s", result["clarify"])
            continue
        documents.append(
            RetrievedDocument(
                text=result.get("text", ""),
                path=result.get("path", ""),
                chunk_index=result.get("chunk_index"),
                score=result.get("score") or result.get("hybrid_score"),
                page=result.get("page"),
                location_percent=result.get("location_percent"),
            )
        )

    return RetrievalResult(query=query, documents=documents)
