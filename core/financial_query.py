from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ENTITY_POSSESSIVE_RE = re.compile(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)'s\b")
_ENTITY_FOR_RE = re.compile(
    r"\b(?:for|of)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\b"
)
_ENTITY_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\b")
_FINANCIAL_TERMS = {
    "finance",
    "financial",
    "tax",
    "expense",
    "expenses",
    "payment",
    "payments",
    "paid",
    "receipt",
    "receipts",
    "invoice",
    "invoices",
    "deductible",
    "deduction",
    "refund",
    "bank",
    "statement",
    "tuition",
}
_EXPENSE_TERMS = {"expense", "expenses", "receipt", "receipts", "invoice", "invoices"}
_PAYMENT_TERMS = {"payment", "payments", "paid", "transfer", "transfers", "refund"}
_ENTITY_STOPWORDS = {
    "what",
    "which",
    "where",
    "when",
    "why",
    "how",
    "who",
    "did",
    "does",
    "do",
    "for",
    "of",
    "in",
    "on",
    "to",
    "tax",
    "taxes",
    "expense",
    "expenses",
    "payment",
    "payments",
}


@dataclass
class FinancialQueryIntent:
    financial_query_mode: bool
    target_entity: Optional[str]
    target_year: Optional[int]
    target_concept: Optional[str]


def _target_year(query: str) -> Optional[int]:
    match = _YEAR_RE.search(query or "")
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _target_entity(query: str) -> Optional[str]:
    match = _ENTITY_POSSESSIVE_RE.search(query or "")
    if match:
        return str(match.group(1) or "").strip() or None
    match = _ENTITY_FOR_RE.search(query or "")
    if match:
        candidate = str(match.group(1) or "").strip()
        if candidate and candidate.lower() not in _ENTITY_STOPWORDS:
            return candidate

    for match in _ENTITY_TOKEN_RE.finditer(query or ""):
        candidate = str(match.group(0) or "").strip()
        if not candidate:
            continue
        if candidate.lower() in _ENTITY_STOPWORDS:
            continue
        return candidate
    return None


def _target_concept(tokens: set[str]) -> Optional[str]:
    if tokens & _EXPENSE_TERMS:
        return "expenses"
    if tokens & _PAYMENT_TERMS:
        return "payments"
    if "tax" in tokens:
        return "tax_relevant_items"
    return None


def detect_financial_query(query: str) -> FinancialQueryIntent:
    tokens = {token.lower() for token in _TOKEN_RE.findall(query or "")}
    financial_query_mode = bool(tokens & _FINANCIAL_TERMS)
    return FinancialQueryIntent(
        financial_query_mode=financial_query_mode,
        target_entity=_target_entity(query) if financial_query_mode else None,
        target_year=_target_year(query) if financial_query_mode else None,
        target_concept=_target_concept(tokens) if financial_query_mode else None,
    )
