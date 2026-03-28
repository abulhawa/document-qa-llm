from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.retrieval.types import RetrievalConfig
from qa_pipeline.types import (
    AnswerContext,
    PromptRequest,
    QueryRewrite,
    RetrievalResult,
    RetrievedDocument,
)

if TYPE_CHECKING:
    from qa_pipeline.coordinator import answer_question as _answer_question_t


def answer_question(*args: Any, **kwargs: Any):
    from qa_pipeline.coordinator import answer_question as _answer_question

    return _answer_question(*args, **kwargs)

__all__ = [
    "answer_question",
    "RetrievalConfig",
    "AnswerContext",
    "PromptRequest",
    "QueryRewrite",
    "RetrievalResult",
    "RetrievedDocument",
]
