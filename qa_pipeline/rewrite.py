import json
from typing import Dict

from config import logger
from core.llm import ask_llm
from qa_pipeline.types import QueryRewrite


def rewrite_question(original_query: str, temperature: float = 0.2) -> QueryRewrite:
    system_prompt = """
    You are an assistant that processes user queries into structured JSON output.

    The user query may be lowercase, lack punctuation, contain typos, or have formatting errors.
    It may also contain typos or spelling errors.
    Do not treat these formatting issues as vagueness, instead, try to fix these errors.
    Evaluate the meaning of the query - not how it's typed.

    Instructions:
    1. If the user intent is clear, complete and specific, return:
    {
        "rewritten": "<keywords or action-oriented phrase>"
    }

    2. If the query is unclear, vague, or lacks context (e.g. uses 'he', 'there', 'it'), request specifics and return:
    {
        "clarify": "<short and polite clarification question>"
    }

    3. Do NOT return full sentences for "rewritten". Only provide useful keywords or compressed task descriptions.
    4. Use phrasing that is short, natural, and similar to what a user might type into a search box.
    5. Avoid overly formal or robotic expressions.
    6. If there are typos or malformed errors, correct these but do not guess the user intent.
    7. Keep clarification questions brief and user-friendly.
    8. Output must be valid JSON only.
    """

    messages: list[Dict[str, str]] = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": original_query.strip()},
    ]

    try:
        rewritten_raw = ask_llm(
            prompt=messages,
            temperature=temperature,
            mode="chat",
            max_tokens=256,
        ).strip()
        rewritten_data = json.loads(rewritten_raw)
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
