"""Use case for document QA."""

from __future__ import annotations

import re
from dataclasses import replace

import qa_pipeline
from app.schemas import DocumentSnippet, QARequest, QAResponse
from qa_pipeline import RetrievalConfig
from config import (
    QA_ENABLE_HYDE,
    QA_ENABLE_QUERY_PLANNING,
    QA_HANDOFF_DYNAMIC_MAX_CHUNKS,
    QA_HANDOFF_DYNAMIC_MIN_CHUNKS,
    QA_HANDOFF_DYNAMIC_RETRIEVAL_TOP_K,
    QA_HANDOFF_DYNAMIC_TOKEN_BUDGET,
    QA_HANDOFF_FIXED_TOP_K,
    QA_HANDOFF_POLICY,
    RETRIEVAL_ENABLE_RERANK,
    RETRIEVAL_RERANK_CANDIDATE_POOL,
    RETRIEVAL_RERANK_TOP_N,
    logger,
)


_Q10_DEBUG_QUERY_TERMS = {"pem", "fuel", "cell", "sliding", "mode", "control"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _resolve_handoff_policy() -> dict[str, int | str | None]:
    policy = str(QA_HANDOFF_POLICY or "dynamic").strip().lower()
    if policy in {"top5", "fixed_top5", "fixed"}:
        return {
            "strategy": "top5",
            "top_k": max(int(QA_HANDOFF_FIXED_TOP_K), 1),
            "dynamic_token_budget": None,
            "dynamic_min_chunks": None,
            "dynamic_max_chunks": None,
        }
    if policy == "dynamic":
        max_chunks = int(QA_HANDOFF_DYNAMIC_MAX_CHUNKS)
        return {
            "strategy": "dynamic",
            "top_k": max(int(QA_HANDOFF_DYNAMIC_RETRIEVAL_TOP_K), 1),
            "dynamic_token_budget": max(int(QA_HANDOFF_DYNAMIC_TOKEN_BUDGET), 1),
            "dynamic_min_chunks": max(int(QA_HANDOFF_DYNAMIC_MIN_CHUNKS), 0),
            "dynamic_max_chunks": max_chunks if max_chunks > 0 else None,
        }

    logger.warning(
        "Unknown QA_HANDOFF_POLICY=%s, defaulting to dynamic policy.",
        policy,
    )
    max_chunks = int(QA_HANDOFF_DYNAMIC_MAX_CHUNKS)
    return {
        "strategy": "dynamic",
        "top_k": max(int(QA_HANDOFF_DYNAMIC_RETRIEVAL_TOP_K), 1),
        "dynamic_token_budget": max(int(QA_HANDOFF_DYNAMIC_TOKEN_BUDGET), 1),
        "dynamic_min_chunks": max(int(QA_HANDOFF_DYNAMIC_MIN_CHUNKS), 0),
        "dynamic_max_chunks": max_chunks if max_chunks > 0 else None,
    }


def _is_q10_debug_question(question: str) -> bool:
    tokens = set(_TOKEN_RE.findall((question or "").lower()))
    if not tokens:
        return False
    return len(tokens & _Q10_DEBUG_QUERY_TERMS) >= 5


def answer(req: QARequest) -> QAResponse:
    """Run the QA pipeline and normalize output for the UI."""
    handoff_policy = _resolve_handoff_policy()
    retrieval_cfg = replace(
        RetrievalConfig(),
        enable_query_planning=QA_ENABLE_QUERY_PLANNING,
        enable_hyde=QA_ENABLE_HYDE,
        enable_rerank=RETRIEVAL_ENABLE_RERANK,
        rerank_top_n=RETRIEVAL_RERANK_TOP_N,
        rerank_candidate_pool=RETRIEVAL_RERANK_CANDIDATE_POOL,
    )
    try:
        context = qa_pipeline.answer_question(
            question=req.question,
            top_k=int(handoff_policy["top_k"]),
            mode=req.mode,
            temperature=req.temperature,
            model=req.model,
            chat_history=req.chat_history,
            retrieval_cfg=retrieval_cfg,
            use_cache=req.use_cache,
            require_grounding=req.require_grounding,
            handoff_strategy=str(handoff_policy["strategy"]),
            handoff_dynamic_token_budget=(
                int(handoff_policy["dynamic_token_budget"])
                if handoff_policy["dynamic_token_budget"] is not None
                else None
            ),
            handoff_dynamic_min_chunks=int(handoff_policy["dynamic_min_chunks"] or 0),
            handoff_dynamic_max_chunks=(
                int(handoff_policy["dynamic_max_chunks"])
                if handoff_policy["dynamic_max_chunks"] is not None
                else None
            ),
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
