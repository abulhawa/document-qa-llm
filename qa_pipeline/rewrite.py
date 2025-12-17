from config import logger
from core.query_rewriter import rewrite_query
from qa_pipeline.types import QueryRewrite


def rewrite_question(original_query: str, temperature: float = 0.2) -> QueryRewrite:
    try:
        rewritten_data = rewrite_query(original_query, temperature=temperature)
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
