from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


_CITATION_MARKER_RE = re.compile(r"\[\s*\d+\s*\]")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


@dataclass(frozen=True)
class GroundingResult:
    score: float
    is_grounded: bool


def _normalize_tokens(text: str) -> set[str]:
    cleaned = _CITATION_MARKER_RE.sub(" ", (text or "").lower())
    return {
        token
        for token in _TOKEN_RE.findall(cleaned)
        if len(token) > 2 and token not in _STOPWORDS
    }


def evaluate_grounding(
    answer: str,
    context_chunks: Sequence[str],
    threshold: float = 0.30,
) -> GroundingResult:
    answer_terms = _normalize_tokens(answer)
    if not answer_terms:
        return GroundingResult(score=0.0, is_grounded=False)

    context_terms: set[str] = set()
    for chunk in context_chunks:
        context_terms.update(_normalize_tokens(chunk))

    if not context_terms:
        return GroundingResult(score=0.0, is_grounded=False)

    overlap = len(answer_terms & context_terms)
    score = round(overlap / len(answer_terms), 4)
    return GroundingResult(score=score, is_grounded=score >= max(0.0, min(1.0, threshold)))
