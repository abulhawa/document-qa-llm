from typing import List, Optional

from config import logger
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

from qa_pipeline.llm_client import generate_answer
from qa_pipeline.prompt_builder import build_prompt
from qa_pipeline.retrieve import retrieve_context
from qa_pipeline.rewrite import rewrite_question
from qa_pipeline.types import AnswerContext


def answer_question(
    question: str,
    top_k: int = 3,
    mode: str = "completion",
    temperature: float = 0.7,
    model: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
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

        # Step 1: Rewrite query or request clarification
        try:
            with start_span("Rewrite Query", TOOL) as rewrite_span:
                rewrite_span.set_attribute(INPUT_VALUE, question)
                rewrite_result = rewrite_question(question, temperature=0.15)
                context.rewritten_question = rewrite_result.rewritten
                context.clarification = rewrite_result.clarify
                rewrite_span.set_attribute(OUTPUT_VALUE, rewrite_result.raw)
                rewrite_span.set_status(STATUS_OK)

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
                retrieval = retrieve_context(context.rewritten_question, top_k)
                context.retrieval = retrieval
                retrieval_span.set_attribute("top_k", top_k)
                retrieval_span.set_attribute("results_found", len(retrieval.documents))
                retrieval_span.set_attribute(OUTPUT_VALUE, retrieval.summary)
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
                )

                llm_span.set_attribute(OUTPUT_VALUE, (context.answer or "")[:1000])
                llm_span.set_status(STATUS_OK)
                logger.info("✅ LLM answered the question.")
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ LLM call failed: %s", exc)
            record_span_error(chain_span, exc)
            context.answer = "❌ LLM call failed."
            return context

        chain_span.set_attribute(OUTPUT_VALUE, context.answer or "")
        chain_span.set_status(STATUS_OK)
        return context
