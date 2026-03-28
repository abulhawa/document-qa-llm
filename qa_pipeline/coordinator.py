from typing import Any, List, Optional

from config import QA_GROUNDING_ENABLED, QA_GROUNDING_THRESHOLD, logger
from core.query_rewriter import has_strong_query_anchors
from core.financial_query import detect_financial_query
from core.retrieval.types import QueryPlan, RetrievalConfig
from tracing import (
    CHAIN,
    INPUT_VALUE,
    LLM,
    OUTPUT_VALUE,
    RETRIEVER,
    STATUS_OK,
    TOOL,
    record_span_error,
    start_span,
)

from qa_pipeline.grounding import evaluate_grounding
from qa_pipeline.handoff import (
    DEFAULT_DYNAMIC_MIN_CHUNKS,
    pack_docs_by_token_budget,
)
from qa_pipeline.llm_client import generate_answer
from qa_pipeline.prompt_builder import build_prompt
from qa_pipeline.retrieve import retrieve_context
from qa_pipeline.rewrite import plan_question, rewrite_question
from qa_pipeline.types import AnswerContext, RetrievalResult


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


def _set_clarify_answer(
    context: AnswerContext,
    chain_span: Any,
    clarification: Optional[str],
) -> bool:
    clarification_text = str(clarification or "").strip()
    if not clarification_text:
        return False
    context.clarification = clarification_text
    context.answer = f"**Clarify**:  \n   -  {clarification_text}.  \n\nTry again!"
    chain_span.set_attribute(OUTPUT_VALUE, context.answer)
    chain_span.set_status(STATUS_OK)
    return True


def answer_question(
    question: str,
    top_k: int = 5,
    mode: str = "completion",
    temperature: float = 0.1,
    model: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
    retrieval_cfg: RetrievalConfig | None = None,
    use_cache: bool = True,
    require_grounding: bool = False,
    handoff_strategy: str = "top5",
    handoff_dynamic_token_budget: Optional[int] = None,
    handoff_dynamic_min_chunks: int = DEFAULT_DYNAMIC_MIN_CHUNKS,
    handoff_dynamic_max_chunks: Optional[int] = None,
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
        chain_span.set_attribute("handoff_strategy", handoff_strategy)

        # Step 1: Rewrite/plan query or request clarification
        planning_enabled = bool(retrieval_cfg.enable_query_planning) if retrieval_cfg else False
        query_plan: QueryPlan | None = None
        try:
            with start_span("Rewrite Query", TOOL) as rewrite_span:
                rewrite_span.set_attribute(INPUT_VALUE, question)
                if planning_enabled:
                    query_plan = plan_question(
                        question,
                        temperature=0.15,
                        use_cache=use_cache,
                        enable_hyde=bool(retrieval_cfg.enable_hyde) if retrieval_cfg else False,
                    )
                    context.rewritten_question = query_plan.semantic_query
                    context.clarification = query_plan.clarify
                    rewrite_span.set_attribute(
                        OUTPUT_VALUE,
                        _as_span_value(
                            {
                                "raw_query": query_plan.raw_query,
                                "semantic_query": query_plan.semantic_query,
                                "bm25_query": query_plan.bm25_query,
                                "hyde_enabled": bool(query_plan.hyde_passage),
                                "clarify": query_plan.clarify or "",
                                "financial_query_mode": query_plan.financial_query_mode,
                                "target_entity": query_plan.target_entity or "",
                                "target_year": query_plan.target_year or "",
                                "target_concept": query_plan.target_concept or "",
                            }
                        ),
                    )
                else:
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

            if not planning_enabled and has_strong_query_anchors(question):
                if (
                    context.rewritten_question
                    and context.rewritten_question.strip() != question.strip()
                ):
                    logger.info(
                        "Rewrite produced alternate text for anchored query; using original question for retrieval."
                    )
                context.rewritten_question = question

            if not planning_enabled and context.clarification and has_strong_query_anchors(question):
                logger.info(
                    "Rewrite requested clarification but query has strong anchors; continuing with exact query."
                )
                context.rewritten_question = question
                context.clarification = None

            if _set_clarify_answer(context, chain_span, context.clarification):
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
                retrieval_query = (
                    query_plan.raw_query
                    if planning_enabled and query_plan is not None
                    else context.rewritten_question
                )
                finance_query_plan: QueryPlan | None = None
                if (
                    not planning_enabled
                    and retrieval_query
                ):
                    financial_intent = detect_financial_query(retrieval_query)
                    if financial_intent.financial_query_mode:
                        finance_query_plan = QueryPlan(
                            raw_query=retrieval_query,
                            semantic_query=retrieval_query,
                            bm25_query=retrieval_query,
                            clarify=None,
                            financial_query_mode=True,
                            target_entity=financial_intent.target_entity,
                            target_year=financial_intent.target_year,
                            target_concept=financial_intent.target_concept,
                        )
                retrieval_span.set_attribute(INPUT_VALUE, retrieval_query)
                if planning_enabled and query_plan is not None:
                    retrieval = retrieve_context(
                        retrieval_query,
                        top_k,
                        retrieval_cfg=retrieval_cfg,
                        query_plan=query_plan,
                    )
                elif finance_query_plan is not None:
                    retrieval = retrieve_context(
                        retrieval_query,
                        top_k,
                        retrieval_cfg=retrieval_cfg,
                        query_plan=finance_query_plan,
                    )
                else:
                    retrieval = retrieve_context(
                        retrieval_query,
                        top_k,
                        retrieval_cfg=retrieval_cfg,
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

        if context.retrieval and _set_clarify_answer(
            context,
            chain_span,
            context.retrieval.clarify,
        ):
            return context

        if not context.retrieval or not context.retrieval.documents:
            logger.warning("⚠️ No relevant results found.")
            context.answer = "No relevant context found to answer the question."
            chain_span.set_attribute(OUTPUT_VALUE, context.answer)
            chain_span.set_status(STATUS_OK)
            return context

        retrieved_documents = list(context.retrieval.documents)
        handoff_policy = str(handoff_strategy or "top5").strip().lower()
        packed_documents = retrieved_documents
        if handoff_policy == "dynamic":
            token_budget = int(handoff_dynamic_token_budget or 0)
            packed_documents, packed_tokens = pack_docs_by_token_budget(
                retrieved_documents,
                token_budget=token_budget,
                min_chunks=handoff_dynamic_min_chunks,
                max_chunks=handoff_dynamic_max_chunks,
            )
            chain_span.set_attribute("handoff_packed_tokens_est", packed_tokens)

        if len(packed_documents) != len(retrieved_documents):
            context.retrieval = RetrievalResult(
                query=context.retrieval.query,
                documents=packed_documents,
                clarify=context.retrieval.clarify,
                stage_metadata=context.retrieval.stage_metadata,
            )
        chain_span.set_attribute("handoff_retrieved_docs", len(retrieved_documents))
        chain_span.set_attribute(
            "handoff_packed_docs",
            len(context.retrieval.documents if context.retrieval else []),
        )

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
