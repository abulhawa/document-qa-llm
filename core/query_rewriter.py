# core/query_rewriter.py

from config import logger
from core.llm import ask_llm
from tracing import start_span, INPUT_VALUE, OUTPUT_VALUE, record_span_error
import json


def rewrite_query(original_query: str, temperature: float = 0.2) -> dict:
    system_prompt = """
    You are an assistant that processes user queries into structured JSON output.
    
    The user query may be lowercase, lack punctuation, contain typos, or have formatting errors.  
    It may also contain typos or spelling errors.  
    Do not treat these formatting issues as vagueness, instead, try to fix these errors.  
    Evaluate the meaning of the query — not how it's typed.

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

    Examples:
    User: where did ali do his bsc studies  
    → { "rewritten": "Ali BSc study location" }

    User: where did he do his bsc studies  
    → { "clarify": "Who are you referring to with 'he'?" }

    User: what about kuwait  
    → { "clarify": "What exactly would you like to know about Kuwait?" }

    User: best time to visit paris  
    → { "rewritten": "best time travel Paris" }
    
    User: what did he say about that topic in that article  
    → { "clarify": "Who is 'he'? And which article or topic are you referring to?" }

    Now process the next user query.
    """

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": original_query.strip()},
    ]

    with start_span("Rewrite Query", kind="TOOL") as span:
        span.set_attribute(INPUT_VALUE, original_query)
        span.set_attribute("rewrite.original_query", original_query)

        try:
            rewritten = ask_llm(
                prompt=messages,
                temperature=temperature,
                mode="chat",
                max_tokens=256,
            ).strip()

            span.set_attribute(OUTPUT_VALUE, rewritten)
            span.set_attribute("rewrite.output_raw", rewritten)

            print("rerewritten:", rewritten)

            rewritten = json.loads(rewritten)

            print("rerewritten json:", rewritten)

            if "clarify" in rewritten:
                span.set_attribute("rewrite.status", "clarify_required")
            elif "rewritten" in rewritten:
                span.set_attribute("rewrite.status", "ok")
            else:
                span.set_attribute("rewrite.status", "unexpected_format")

            return rewritten

        except Exception as e:
            record_span_error(span, e)
            logger.exception("Query rewriting failed")
            return {"Error": "Query rewriting failed. Please try again or rephrase."}
