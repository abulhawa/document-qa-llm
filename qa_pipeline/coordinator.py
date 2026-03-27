from typing import Any, List, Optional

from config import logger, QA_GROUNDING_ENABLED, QA_GROUNDING_THRESHOLD
from core.query_rewriter import has_strong_query_anchors
from core.retrieval.types import RetrievalConfig
from tracing import (
    start_span,
    record_span_error,
    STATUS_OK,
    RETRIEVER,
    LLM,
    INPUT_VALUE,
    OUTPUT_VALUE,
    CHAIN,
    TOOL,
)

from qa_pipeline.grounding import evaluate_grounding
from qa_pipeline.llm_client import generate_answer
from qa_pipeline.prompt_builder import build_prompt
from qa_pipeline.retrieve import retrieve_context
from qa_pipeline.rewrite import rewrite_question
from qa_pipeline.types import AnswerContext


AttributePrimitive = str | bool | int | float
SpanAttributeValue = AttributePrimitive | list[str]


def _as_span_value(value: Any) -> SpanAttributeValue:
    """
    Coerce arbitrary values into the limited set of types
    allowed by OpenTelemetry span attributes.
    """
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value)


def answer_question(
    question: str,
    top_k: int = 3,
    mode: str = "completion",
    temperature: float = 0.1,
    model: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
    retrieval_cfg: RetrievalConfig | None = None,
    use_cache: bool = True,
    require_grounding: bool = False,
) -> AnswerContext:
    """Orchestrate the QA pipeline to answer a question."""

    context = AnswerContext(
        question=question,
        mode=mode,
        temperature=temperature,
        model=model,
        chat_history=chat_history or [],
    )

    with start_span("QA chain", CHAIN) as chain_span:
        chain_span.set_attribute(INPUT_VALUE, question)
        chain_span.set_attribute("mode", mode)
        chain_span.set_attribute("top_k", top_k)
        chain_span.set_attribute("temperature", temperature)
        chain_span.set_attribute("model", model or "unknown")
        chain_span.set_attribute("require_grounding", require_grounding)

        # Step 1: Rewrite query or request clarification
        try:
            with start_span("Rewrite Query", TOOL) as rewrite_span:
                rewrite_span.set_attribute(INPUT_VALUE, question)
                rewrite_result = rewrite_question(
                    question,
                    temperature=0.15,
                    use_cache=use_cache,
                )
                context.rewritten_question = rewrite_result.rewritten
                context.clarification = rewrite_result.clarify
                rewrite_span.set_attribute(
                    OUTPUT_VALUE, _as_span_value(rewrite_result.raw)
                )
                rewrite_span.set_status(STATUS_OK)

            if has_strong_query_anchors(question):
                if (
                    context.rewritten_question
                    and context.rewritten_question.strip() != question.strip()
                ):
                    logger.info(
                        "Rewrite produced alternate text for anchored query; using original question for retrieval."
                    )
                context.rewritten_question = question

            if context.clarification and has_strong_query_anchors(question):
                logger.info(
                    "Rewrite requested clarification but query has strong anchors; continuing with exact query."
                )
                context.rewritten_question = question
                context.clarification = None

            if context.clarification:
                context.answer = (
                    f"**Clarify**:  \n   -  {context.clarification}.  \n\nTry again!"
                )
                chain_span.set_attribute(OUTPUT_VALUE, context.answer)
                chain_span.set_status(STATUS_OK)
                return context

            if not context.rewritten_question:
                context.answer = "❌ Unexpected error occurred... ERR-QRWR"
                chain_span.set_attribute(OUTPUT_VALUE, context.answer)
                return context
        except Exception as exc:  # noqa: BLE001
            record_span_error(chain_span, exc)
            context.answer = "❌ Unexpected error occurred... ERR-QRWR"
            return context

        # Step 2: Retrieve context
        try:
            with start_span("Retriever", RETRIEVER) as retrieval_span:
                retrieval_span.set_attribute(INPUT_VALUE, context.rewritten_question)
                retrieval = retrieve_context(
                    context.rewritten_question, top_k, retrieval_cfg=retrieval_cfg
                )
                context.retrieval = retrieval
                retrieval_span.set_attribute("top_k", top_k)
                retrieval_span.set_attribute("results_found", len(retrieval.documents))
                retrieval_span.set_attribute(
                    OUTPUT_VALUE, _as_span_value(retrieval.summary)
                )
                retrieval_span.set_status(STATUS_OK)
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ Retrieval failed: %s", exc)
            record_span_error(chain_span, exc)
            context.answer = "❌ Retrieval failed."
            return context

        if not context.retrieval or not context.retrieval.documents:
            logger.warning("⚠️ No relevant results found.")
            context.answer = "No relevant context found to answer the question."
            chain_span.set_attribute(OUTPUT_VALUE, context.answer)
            chain_span.set_status(STATUS_OK)
            return context

        # Step 3: Build prompt
        context.prompt_request = build_prompt(
            context.retrieval, question, mode=mode, chat_history=context.chat_history
        )

        # Step 4: Ask the LLM
        try:
            with start_span("LLM", LLM) as llm_span:
                llm_span.set_attribute("model", model or "unknown")
                llm_span.set_attribute("temperature", temperature)
                llm_span.set_attribute("mode", mode)
                llm_span.set_attribute(
                    INPUT_VALUE, str(context.prompt_request.prompt)[:2000]
                )

                context.answer = generate_answer(
                    prompt_request=context.prompt_request,
                    temperature=temperature,
                    model=model,
                    use_cache=use_cache,
                )

                llm_span.set_attribute(OUTPUT_VALUE, (context.answer or "")[:1000])
                llm_span.set_status(STATUS_OK)
                logger.info("✅ LLM answered the question.")
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ LLM call failed: %s", exc)
            record_span_error(chain_span, exc)
            context.answer = "❌ LLM call failed."
            return context

        should_check_grounding = QA_GROUNDING_ENABLED or require_grounding
        if should_check_grounding:
            try:
                grounding = evaluate_grounding(
                    answer=context.answer or "",
                    context_chunks=context.retrieval.context_chunks,
                    threshold=QA_GROUNDING_THRESHOLD,
                )
                context.grounding_score = grounding.score
                context.is_grounded = grounding.is_grounded
                chain_span.set_attribute("grounding_score", grounding.score)
                chain_span.set_attribute("is_grounded", grounding.is_grounded)
                if require_grounding and not grounding.is_grounded:
                    context.answer = "I don't know."
            except Exception as exc:  # noqa: BLE001
                logger.warning("Grounding check failed: %s", exc)
                if require_grounding:
                    context.grounding_score = 0.0
                    context.is_grounded = False
                    chain_span.set_attribute("grounding_score", 0.0)
                    chain_span.set_attribute("is_grounded", False)
                    context.answer = "I don't know."

        chain_span.set_attribute(OUTPUT_VALUE, context.answer or "")
        chain_span.set_status(STATUS_OK)
        return context
