from typing import List, Optional

from config import logger
from core.retrieval.pipeline import retrieve
from core.retrieval.types import QueryPlan, RetrievalConfig, RetrievalDeps
from core.retrieval.reranker import build_configured_reranker
from core.embeddings import embed_texts
from core.vector_store import retrieve_top_k as semantic_retriever
from core.opensearch_store import search as keyword_retriever, fetch_sibling_chunks
from qa_pipeline.types import RetrievalResult, RetrievedDocument


def default_retrieval_config(top_k: int) -> RetrievalConfig:
    return RetrievalConfig(top_k=top_k)


def default_retrieval_deps() -> RetrievalDeps:
    return RetrievalDeps(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
        embed_texts=embed_texts,
        cross_encoder=build_configured_reranker(),
        sibling_chunk_fetcher=fetch_sibling_chunks,
    )


def retrieve_context(
    query: str,
    top_k: int,
    retrieval_cfg: Optional[RetrievalConfig] = None,
    deps: Optional[RetrievalDeps] = None,
    query_plan: Optional[QueryPlan] = None,
) -> RetrievalResult:
    cfg = retrieval_cfg.with_top_k(top_k) if retrieval_cfg else default_retrieval_config(top_k)
    deps = deps or default_retrieval_deps()

    logger.info("🔍 Running retrieval for user question...")
    if query_plan is not None:
        output = retrieve(query, cfg=cfg, deps=deps, query_plan=query_plan)
    else:
        output = retrieve(query, cfg=cfg, deps=deps)

    documents: List[RetrievedDocument] = []
    output_stage_metadata = getattr(output, "stage_metadata", None) or {}
    if output.clarify:
        logger.info("Retriever requested clarification: %s", output.clarify)
        return RetrievalResult(
            query=query,
            documents=documents,
            clarify=output.clarify,
            stage_metadata=output_stage_metadata,
        )

    for result in output.documents:
        retrieval_score = result.get("retrieval_score")
        documents.append(
            RetrievedDocument(
                text=result.get("text", ""),
                path=result.get("path", ""),
                chunk_index=result.get("chunk_index"),
                score=retrieval_score if retrieval_score is not None else result.get("score"),
                page=result.get("page"),
                location_percent=result.get("location_percent"),
                doc_type=result.get("doc_type"),
                person_name=result.get("person_name"),
                authority_rank=result.get("authority_rank"),
                checksum=result.get("checksum"),
                query_variant=result.get("_query_variant"),
                query_text=result.get("_query_text"),
                query_channel=result.get("_query_channel"),
                source_family=result.get("_financial_source_family"),
                is_financial_document=result.get("is_financial_document"),
                document_date=result.get("document_date"),
                mentioned_years=result.get("mentioned_years"),
                transaction_dates=result.get("transaction_dates"),
                tax_years_referenced=result.get("tax_years_referenced"),
                financial_record_type=result.get("financial_record_type"),
                financial_metadata_source=result.get("financial_metadata_source"),
                financial_retrieval_stage=result.get("_financial_retrieval_stage"),
            )
        )

    return RetrievalResult(
        query=query,
        documents=documents,
        stage_metadata=output_stage_metadata,
    )
