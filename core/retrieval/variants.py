from __future__ import annotations
from typing import List, Tuple, Dict, Any
from config import logger
from core.query_rewriter import rewrite_query

Variant = Tuple[str, float | None]  # (query_text, bm25_weight)


def generate_variants(original: str, *, rewrite_temp: float = 0.15) -> Dict[str, Any]:
    """
    Returns:
      - {"clarify": "..."}                       -> caller should handle
      - {"variants": [(exact,1.0),(rewritten,0.6)], "rewritten": "..."}
    """
    result = rewrite_query(original, temperature=rewrite_temp)
    if "clarify" in result:
        return {"clarify": result["clarify"]}

    rewritten = result.get("rewritten", "").strip()
    exact = original.strip()

    variants: List[Variant] = [(exact, 1.0)]
    if rewritten and rewritten.lower() != exact.lower():
        variants.append((rewritten, 0.6))
    else:
        logger.info("Rewriter produced no effective change; using exact query only.")
    return {"variants": variants, "rewritten": rewritten or exact}
