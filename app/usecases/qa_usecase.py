"""Use case for document QA."""

from __future__ import annotations

import re
from dataclasses import replace

import qa_pipeline
from app.schemas import DocumentSnippet, QARequest, QAResponse
from qa_pipeline import RetrievalConfig
from config import (
    RETRIEVAL_ENABLE_RERANK,
    RETRIEVAL_RERANK_CANDIDATE_POOL,
    RETRIEVAL_RERANK_TOP_N,
    logger,
)


_Q10_DEBUG_QUERY_TERMS = {"pem", "fuel", "cell", "sliding", "mode", "control"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _is_q10_debug_question(question: str) -> bool:
    tokens = set(_TOKEN_RE.findall((question or "").lower()))
    if not tokens:
        return False
    return len(tokens & _Q10_DEBUG_QUERY_TERMS) >= 5


def answer(req: QARequest) -> QAResponse:
    """Run the QA pipeline and normalize output for the UI."""
    retrieval_cfg = replace(
        RetrievalConfig(),
        enable_rerank=RETRIEVAL_ENABLE_RERANK,
        rerank_top_n=RETRIEVAL_RERANK_TOP_N,
        rerank_candidate_pool=RETRIEVAL_RERANK_CANDIDATE_POOL,
    )
    try:
        context = qa_pipeline.answer_question(
            question=req.question,
            mode=req.mode,
            temperature=req.temperature,
            model=req.model,
            chat_history=req.chat_history,
            retrieval_cfg=retrieval_cfg,
            use_cache=req.use_cache,
            require_grounding=req.require_grounding,
        )
    except Exception as exc:  # noqa: BLE001
        return QAResponse(answer="", error=str(exc))

    documents = []
    sources = []
    if context.retrieval:
        sources = context.retrieval.sources
        documents = [
            DocumentSnippet(
                text=doc.text,
                path=doc.path,
                chunk_index=doc.chunk_index,
                score=doc.score,
                page=doc.page,
                location_percent=doc.location_percent,
            )
            for doc in context.retrieval.documents
        ]
        if _is_q10_debug_question(req.question):
            logger.info(
                "Q10 debug | final_ui_sources=%s | final_ui_documents=%s",
                sources,
                [doc.path for doc in documents],
            )

    return QAResponse(
        answer=context.answer or "",
        sources=sources,
        documents=documents,
        rewritten_question=context.rewritten_question,
        clarification=context.clarification,
        is_grounded=context.is_grounded,
        grounding_score=context.grounding_score,
    )
