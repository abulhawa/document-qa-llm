"""Use case for document QA."""

from __future__ import annotations

import qa_pipeline
from app.schemas import DocumentSnippet, QARequest, QAResponse
from qa_pipeline import RetrievalConfig


def answer(req: QARequest) -> QAResponse:
    """Run the QA pipeline and normalize output for the UI."""
    retrieval_cfg = RetrievalConfig()
    try:
        context = qa_pipeline.answer_question(
            question=req.question,
            mode=req.mode,
            temperature=req.temperature,
            model=req.model,
            chat_history=req.chat_history,
            retrieval_cfg=retrieval_cfg,
            use_cache=req.use_cache,
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

    return QAResponse(
        answer=context.answer or "",
        sources=sources,
        documents=documents,
        rewritten_question=context.rewritten_question,
        clarification=context.clarification,
    )
