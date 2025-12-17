from typing import List, Optional

from config import logger
from core.retrieval.pipeline import retrieve
from core.retrieval.types import RetrievalConfig, RetrievalDeps
from core.embeddings import embed_texts
from core.vector_store import retrieve_top_k as semantic_retriever
from core.opensearch_store import search as keyword_retriever
from qa_pipeline.types import RetrievalResult, RetrievedDocument


def default_retrieval_config(top_k: int) -> RetrievalConfig:
    return RetrievalConfig(top_k=top_k)


def default_retrieval_deps() -> RetrievalDeps:
    return RetrievalDeps(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
        embed_texts=embed_texts,
        cross_encoder=None,
    )


def retrieve_context(
    query: str,
    top_k: int,
    retrieval_cfg: Optional[RetrievalConfig] = None,
    deps: Optional[RetrievalDeps] = None,
) -> RetrievalResult:
    cfg = retrieval_cfg.with_top_k(top_k) if retrieval_cfg else default_retrieval_config(top_k)
    deps = deps or default_retrieval_deps()

    logger.info("🔍 Running retrieval for user question...")
    output = retrieve(query, cfg=cfg, deps=deps)

    documents: List[RetrievedDocument] = []
    if output.clarify:
        logger.info("Retriever requested clarification: %s", output.clarify)
        return RetrievalResult(query=query, documents=documents)

    for result in output.documents:
        documents.append(
            RetrievedDocument(
                text=result.get("text", ""),
                path=result.get("path", ""),
                chunk_index=result.get("chunk_index"),
                score=result.get("score") or result.get("retrieval_score"),
                page=result.get("page"),
                location_percent=result.get("location_percent"),
            )
        )

    return RetrievalResult(query=query, documents=documents)
