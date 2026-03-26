from typing import Any
import re
from config import logger
from core.llm import ask_llm
import json


_ANCHOR_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")
_QUOTE_RE = re.compile(r"['\"][^'\"]{3,}['\"]")
_AMBIGUOUS_PRONOUNS = {"he", "she", "they", "it", "him", "her", "them"}
_STRONG_ANCHOR_TERMS = {
    "cv",
    "resume",
    "cover",
    "letter",
    "reference",
    "report",
    "paper",
    "course",
    "lecture",
    "contract",
    "policy",
    "invoice",
    "receipt",
    "jobcenter",
    "form",
    "formular",
    "insurance",
    "tax",
    "phd",
    "msc",
    "bsc",
}


def has_strong_query_anchors(original_query: str) -> bool:
    query = (original_query or "").strip()
    if not query:
        return False

    if _QUOTE_RE.search(query):
        return True

    lowered_tokens = [tok.lower() for tok in _ANCHOR_TOKEN_RE.findall(query)]
    if any(tok.isdigit() and len(tok) >= 4 for tok in lowered_tokens):
        return True
    if any(tok in _STRONG_ANCHOR_TERMS for tok in lowered_tokens):
        if not any(tok in _AMBIGUOUS_PRONOUNS for tok in lowered_tokens):
            return True

    # Possessive or hyphenated anchors often indicate concrete entities/titles.
    if re.search(r"\b[a-z]{2,}'s\b", query, flags=re.IGNORECASE):
        return True
    if re.search(r"\b[a-z0-9]+-[a-z0-9]+\b", query, flags=re.IGNORECASE):
        return True

    # Require stronger lexical signal when unresolved pronouns are present.
    if any(tok in _AMBIGUOUS_PRONOUNS for tok in lowered_tokens):
        return False

    long_tokens = [tok for tok in lowered_tokens if len(tok) >= 7 and tok.isalpha()]
    return len(long_tokens) >= 2


def rewrite_query(
    original_query: str,
    temperature: float = 0.2,
    use_cache: bool = True,
) -> dict[str, Any]:
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
    8. If the query has strong anchors (named entity, document type/title, specific year/number), do not ask for clarification; return a rewritten query instead.
    9. Output must be valid JSON only.

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

    try:
        rewritten = ask_llm(
            prompt=messages,
            temperature=temperature,
            mode="chat",
            max_tokens=256,
            use_cache=use_cache,
        ).strip()

        rewritten = json.loads(rewritten)
        if not isinstance(rewritten, dict):
            raise ValueError("Rewrite output is not a JSON object")
        if "clarify" in rewritten and has_strong_query_anchors(original_query):
            logger.info("Rewrite clarification bypassed due to strong query anchors.")
            return {"rewritten": original_query.strip()}
        return rewritten

    except Exception:
        logger.exception("Query rewriting failed")
        return {"Error": "Query rewriting failed. Please try again or rephrase."}
