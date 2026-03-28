from config import logger
from core.query_rewriter import build_query_plan, rewrite_query
from core.retrieval.types import QueryPlan
from qa_pipeline.types import QueryRewrite


def rewrite_question(
    original_query: str,
    temperature: float = 0.2,
    use_cache: bool = True,
) -> QueryRewrite:
    try:
        rewritten_data = rewrite_query(
            original_query,
            temperature=temperature,
            use_cache=use_cache,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query rewriting failed", exc_info=exc)
        return QueryRewrite(raw={"Error": "Query rewriting failed."})

    query_rewrite = QueryRewrite(raw=rewritten_data)
    if isinstance(rewritten_data, dict):
        if "clarify" in rewritten_data:
            query_rewrite.clarify = rewritten_data["clarify"]
        elif "rewritten" in rewritten_data:
            query_rewrite.rewritten = rewritten_data["rewritten"]
        else:
            logger.warning("Unexpected rewrite format: %s", rewritten_data)
    else:
        logger.warning("Rewrite result was not a dict: %s", rewritten_data)

    return query_rewrite


def plan_question(
    original_query: str,
    *,
    temperature: float = 0.15,
    use_cache: bool = True,
    enable_hyde: bool = False,
) -> QueryPlan:
    try:
        return build_query_plan(
            original_query,
            temperature=temperature,
            use_cache=use_cache,
            enable_hyde=enable_hyde,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query planning failed", exc_info=exc)
        query = (original_query or "").strip()
        return QueryPlan(
            raw_query=query,
            semantic_query=query,
            bm25_query=query,
            hyde_passage=None,
            clarify=None,
        )
