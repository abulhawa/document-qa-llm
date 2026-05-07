"""
ingestion/financial_extractor.py
================================
Responsible for detecting whether a document is financially relevant and, if so,
extracting structured financial metadata and individual transaction records from it.

This module is called by the ingestion orchestrator (orchestrator.py) AFTER
doc_classifier.py has already identified the document type (doc_type). It uses
the doc_type as a signal to decide how aggressively to run financial extraction.

High-level flow
---------------
1. _source_family()        — maps (path, doc_type, text) → a family label
                             e.g. "tax_document", "publication", "cv"
                             This is the gating decision: suppressed families
                             (books, CVs, research papers) skip extraction.

2. Deterministic extraction — regex-based passes over the full text and each
                             chunk to pull out dates, amounts, counterparties,
                             and tax signals. Produces a list of candidate records.

3. LLM fallback            — if the document looks financial but the deterministic
                             pass came back thin (missing dates, counterparties, etc.),
                             an LLM call is made with a structured JSON schema prompt
                             to fill the gaps.

4. merge_duplicate_records() — deduplicates records by a composite merge key
                             (type + date + amount + currency + counterparty).

5. Returns FinancialExtractionResult with:
   - document_metadata: flat dict merged into every chunk and the full-text doc
   - records: list of individual transaction records upserted to financial_records index
   - source_family, used_llm_fallback, fallback_reason

Known issue / design note
--------------------------
_source_family() has a catch-all rule at the end:
    if _TAX_SIGNAL_RE.search(sample): return "official_letter"
This runs AFTER the doc_type checks, but _TAX_SIGNAL_RE is broad (matches "cost",
"expense", "receipt") and can fire on engineering papers, academic texts, etc.
The correct fix is to guard with a suppression check BEFORE running financial
extraction, or move the doc_type → suppressed_family check earlier in _source_family.
See extract_financial_enrichment() for the recommended suppression guard location.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Sequence

from config import logger
from core.llm import ask_llm


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
# Matches 4-digit years in the range 1900–2099.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_DATE_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b|"
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4}\b",
    re.IGNORECASE,
)
# Matches monetary amounts with optional currency symbols/codes on either side.
# Handles European (1.234,56) and US (1,234.56) number formats.
_AMOUNT_RE = re.compile(
    r"(?P<currency_1>EUR|USD|GBP|CHF|€|\$|£)\s*(?P<amount_1>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)|"
    r"(?P<amount_2>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*(?P<currency_2>EUR|USD|GBP|CHF|€|\$|£)",
    re.IGNORECASE,
)
# Extracts the name following keywords like "payee:", "to:", "from:", "vendor:".
_COUNTERPARTY_RE = re.compile(
    r"(?:payee|merchant|vendor|recipient|to|from)\s*[:\-]?\s*([A-Z][A-Za-z0-9&.,'() \-]{2,80})",
    re.IGNORECASE,
)
# Broad keyword match for financial relevance signals.
# WARNING: This is intentionally broad — words like "receipt", "invoice", "expense"
# appear in non-financial documents too (engineering papers, academic texts).
# Do not use this alone as the sole gate for financial classification.
_TAX_SIGNAL_RE = re.compile(
    r"\b(tax|vat|deductible|expense|receipt|invoice|tuition|school fee|medical)\b",
    re.IGNORECASE,
)
_EXPENSE_CATEGORY_RULES = (
    ("education", re.compile(r"\b(tuition|school|university|course fee)\b", re.IGNORECASE)),
    ("medical", re.compile(r"\b(medical|clinic|hospital|doctor|pharmacy)\b", re.IGNORECASE)),
    ("travel", re.compile(r"\b(travel|flight|train|bus|ticket|hotel)\b", re.IGNORECASE)),
    ("insurance", re.compile(r"\b(insurance|premium)\b", re.IGNORECASE)),
    ("housing", re.compile(r"\b(rent|lease|housing|utilities?)\b", re.IGNORECASE)),
    ("professional", re.compile(r"\b(software|subscription|equipment|office)\b", re.IGNORECASE)),
)

# Document families that are actively processed for financial records.
# If _source_family() returns one of these, extraction proceeds fully.
_PREFERRED_FINANCIAL_FAMILIES = {
    "tax_document",
    "bank_statement",
    "receipt",
    "invoice",
    "payment_confirmation",
    "school_fee_letter",
    "official_letter",
}

# Document families where financial extraction is explicitly suppressed.
# Even if the text contains amounts or dates, these are not financial documents.
# IMPORTANT: _source_family() must resolve to one of these BEFORE the
# _TAX_SIGNAL_RE catch-all fires, otherwise suppression is bypassed.
_SUPPRESSED_FAMILIES = {
    "book",
    "course_material",
    "publication",
    "cv",
    "reference",
    "archive_misc",
}


def preferred_financial_families() -> set[str]:
    return set(_PREFERRED_FINANCIAL_FAMILIES)


def suppressed_financial_families() -> set[str]:
    return set(_SUPPRESSED_FAMILIES)


@dataclass
class FinancialExtractionResult:
    document_metadata: Dict[str, Any]
    records: List[Dict[str, Any]]
    source_family: str
    used_llm_fallback: bool
    fallback_reason: Optional[str] = None


def _normalize_currency(raw: str) -> str:
    upper = str(raw or "").strip().upper()
    if upper == "€":
        return "EUR"
    if upper == "$":
        return "USD"
    if upper == "£":
        return "GBP"
    return upper or "UNKNOWN"


def _normalize_amount(raw: str) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        pieces = text.split(",")
        if len(pieces[-1]) in {1, 2}:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _parse_date_token(token: str) -> Optional[str]:
    text = str(token or "").strip()
    if not text:
        return None
    formats = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%d.%m.%Y",
        "%d.%m.%y",
        "%B %d %Y",
        "%b %d %Y",
        "%B %d, %Y",
        "%b %d, %Y",
    )
    cleaned = re.sub(r"\s+", " ", text).strip()
    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return None


def _extract_dates(text: str) -> List[str]:
    values = []
    for match in _DATE_RE.finditer(text or ""):
        parsed = _parse_date_token(match.group(0))
        if parsed:
            values.append(parsed)
    return _sorted_unique(values)


def _extract_years(text: str) -> List[int]:
    out: list[int] = []
    for match in _YEAR_RE.finditer(text or ""):
        try:
            out.append(int(match.group(0)))
        except ValueError:
            continue
    return sorted(set(out))


def _extract_amount_matches(text: str) -> List[Dict[str, Any]]:
    matches: list[Dict[str, Any]] = []
    for match in _AMOUNT_RE.finditer(text or ""):
        currency = match.group("currency_1") or match.group("currency_2")
        raw_amount = match.group("amount_1") or match.group("amount_2")
        amount = _normalize_amount(raw_amount)
        if amount is None:
            continue
        span_text = match.group(0)
        matches.append(
            {
                "amount": amount,
                "currency": _normalize_currency(currency or ""),
                "span": span_text,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return matches


def _extract_counterparty(text: str) -> Optional[str]:
    match = _COUNTERPARTY_RE.search(text or "")
    if not match:
        return None
    value = re.sub(r"\s+", " ", (match.group(1) or "").strip())
    value = value.strip(" ,.;:")
    return value or None


def _extract_tax_signals(text: str) -> List[str]:
    signals: list[str] = []
    for match in _TAX_SIGNAL_RE.finditer(text or ""):
        signal = str(match.group(1) or "").lower().strip()
        if signal and signal not in signals:
            signals.append(signal)
    return signals


def _derive_expense_category(text: str) -> Optional[str]:
    for name, pattern in _EXPENSE_CATEGORY_RULES:
        if pattern.search(text or ""):
            return name
    return None


def _sorted_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return sorted(out)


def _limited_unique(values: Iterable[Any], limit: int = 20) -> List[Any]:
    out = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _source_family(path: str, doc_type: Optional[str], full_text: str) -> str:
    """
    Map a document to a source family label used to gate and guide financial extraction.

    Priority order (first match wins):
    1. Path/filename keywords (most reliable — e.g. "invoice", "kontoauszug")
    2. doc_type from doc_classifier (e.g. "research_paper" → "publication")
    3. Text-sample heuristics (e.g. presence of IBAN → "bank_statement")
    4. _TAX_SIGNAL_RE catch-all → "official_letter" (LAST RESORT — broad, can misfire)
    5. Default fallback → "archive_misc"

    KNOWN BUG: The _TAX_SIGNAL_RE catch-all (step 4) runs after some doc_type
    checks but can override them if the doc_type check is positioned too late.
    For example, a research_paper with the word "expense" in it can be
    re-labelled as "official_letter", bypassing suppression.
    Fix: ensure doc_type → suppressed family checks come BEFORE step 4.
    """
    p = (path or "").lower()
    d = (doc_type or "").strip().lower()
    sample = (full_text or "")[:5000].lower()

    if d in {"invoice"} or "invoice" in p:
        return "invoice"
    if "receipt" in p or "quittung" in p:
        return "receipt"
    if "bank statement" in p or "kontoauszug" in p or "iban" in sample:
        return "bank_statement"
    if "payment confirmation" in p or "payment received" in sample:
        return "payment_confirmation"
    if "tax" in p or "steuer" in p or "tax return" in sample:
        return "tax_document"
    if "tuition" in p or "school fee" in sample or "semester fee" in sample:
        return "school_fee_letter"
    if d in {"government_form", "insurance_letter", "payroll"}:
        return "official_letter"
    if d in {"course_material"}:
        return "course_material"
    if d in {"research_paper", "technical_report"}:
        return "publication"
    if d in {"cv", "resume"}:
        return "cv"
    if d in {"reference_letter"}:
        return "reference"
    if re.search(r"\b(book|ebook)\b", p):
        return "book"
    if _TAX_SIGNAL_RE.search(sample):
        return "official_letter"
    return "archive_misc"


def _build_merge_key(record: Dict[str, Any]) -> str:
    parts = [
        str(record.get("record_type") or "").lower().strip(),
        str(record.get("date") or "").strip(),
        f"{float(record.get('amount') or 0.0):.2f}",
        str(record.get("currency") or "").upper().strip(),
        re.sub(r"\s+", " ", str(record.get("counterparty") or "").lower()).strip(),
    ]
    return "|".join(parts)


def _default_record_type(source_family: str, text: str) -> str:
    """
    Assign a default financial record type when no explicit type was extracted.
    Checks text for 'refund' or 'tax' first, then falls back to family-based rules.
    """
    lowered = (text or "").lower()
    if "refund" in lowered:
        return "refund"
    if "tax" in lowered:
        return "tax_payment"
    if source_family in {"receipt", "invoice", "school_fee_letter"}:
        return "expense"
    if source_family in {"payment_confirmation", "bank_statement"}:
        return "payment"
    return "financial_event"


def _record_confidence(*, has_date: bool, has_counterparty: bool, llm: bool) -> float:
    """
    Score extraction confidence 0.0-1.0 based on signal quality.
    LLM-extracted records are capped at 0.6 (less reliable than deterministic).
    Deterministic records score higher when both date and counterparty are present.
    """
    if llm:
        return 0.6
    if has_date and has_counterparty:
        return 0.8
    if has_date:
        return 0.72
    return 0.55


def _record_is_valid(record: Dict[str, Any]) -> bool:
    try:
        amount = float(record.get("amount") or 0.0)
    except (TypeError, ValueError):
        return False
    date_text = str(record.get("date") or "").strip()
    currency = str(record.get("currency") or "").strip()
    if amount <= 0:
        return False
    if not date_text or _parse_date_token(date_text) is None:
        return False
    if not currency:
        return False
    return True


def merge_duplicate_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate financial records using a composite merge key:
        record_type | date | amount | currency | counterparty

    When duplicates are found (same key), the canonical record accumulates
    source_links from all duplicates and takes the highest confidence score.
    If duplicates came from different extraction methods (deterministic vs llm),
    the method is set to "hybrid".
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        key = str(record.get("merge_key") or _build_merge_key(record))
        if not key:
            continue
        record["merge_key"] = key
        record_links = list(record.get("source_links") or [])
        if key not in merged:
            canonical = dict(record)
            canonical["source_links"] = _limited_unique(record_links, limit=50)
            canonical["source_count"] = len(canonical["source_links"])
            merged[key] = canonical
            continue

        existing = merged[key]
        combined_links = _limited_unique(
            list(existing.get("source_links") or []) + record_links,
            limit=50,
        )
        existing["source_links"] = combined_links
        existing["source_count"] = len(combined_links)
        existing["confidence"] = max(
            float(existing.get("confidence") or 0.0),
            float(record.get("confidence") or 0.0),
        )
        if str(existing.get("extraction_method") or "") != str(record.get("extraction_method") or ""):
            existing["extraction_method"] = "hybrid"
    return list(merged.values())


def _build_llm_prompt(path: str, text: str, source_family: str) -> List[Dict[str, str]]:
    schema = {
        "document_metadata": {
            "document_date": "YYYY-MM-DD or null",
            "mentioned_years": [2022],
            "transaction_dates": ["YYYY-MM-DD"],
            "tax_years_referenced": [2022],
            "amounts": [123.45],
            "counterparties": ["Vendor Name"],
            "tax_relevance_signals": ["tax", "invoice"],
            "expense_category": "education or null",
            "financial_record_type": "expense or payment",
        },
        "records": [
            {
                "record_type": "expense",
                "date": "YYYY-MM-DD",
                "amount": 123.45,
                "currency": "EUR",
                "counterparty": "Vendor Name",
                "description": "Short evidence description",
                "source_text_span": "short span from source text",
            }
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Extract structured financial metadata and records as JSON only. "
                "Do not include markdown or explanations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Path: {path}\n"
                f"Source family hint: {source_family}\n"
                f"Schema: {json.dumps(schema, ensure_ascii=True)}\n"
                f"Document text:\n{text[:8000]}"
            ),
        },
    ]


def _try_llm_fallback(path: str, text: str, source_family: str) -> Dict[str, Any]:
    prompt = _build_llm_prompt(path, text, source_family)
    try:
        raw = ask_llm(
            prompt=prompt,
            mode="chat",
            temperature=0.0,
            max_tokens=1200,
            use_cache=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Financial LLM fallback failed: %s", exc)
        return {}

    payload = str(raw or "").strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        payload = payload.replace("json", "", 1).strip()
    try:
        parsed = json.loads(payload)
    except Exception:  # noqa: BLE001
        logger.warning("Financial LLM fallback returned invalid JSON.")
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _normalize_llm_records(
    records: Sequence[Dict[str, Any]],
    *,
    document_id: str,
    checksum: str,
    source_family: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(records):
        date_text = _parse_date_token(str(raw.get("date") or ""))
        amount = _normalize_amount(str(raw.get("amount") or ""))
        if date_text is None or amount is None:
            continue
        currency = _normalize_currency(str(raw.get("currency") or ""))
        if not currency:
            continue
        counterparty = str(raw.get("counterparty") or "").strip() or None
        record = {
            "record_type": str(raw.get("record_type") or "financial_event").strip().lower(),
            "date": date_text,
            "amount": amount,
            "currency": currency,
            "counterparty": counterparty,
            "description": str(raw.get("description") or "").strip(),
            "confidence": _record_confidence(
                has_date=True,
                has_counterparty=bool(counterparty),
                llm=True,
            ),
            "document_id": document_id,
            "checksum": checksum,
            "chunk_id": None,
            "extraction_method": "llm",
            "source_text_span": str(raw.get("source_text_span") or "").strip(),
            "financial_record_version": "v1",
            "source_family": source_family,
            "year": int(date_text[:4]),
            "source_links": [
                {
                    "document_id": document_id,
                    "checksum": checksum,
                    "chunk_id": None,
                    "source_text_span": str(raw.get("source_text_span") or "").strip(),
                    "extraction_method": "llm",
                    "confidence": _record_confidence(
                        has_date=True,
                        has_counterparty=bool(counterparty),
                        llm=True,
                    ),
                }
            ],
        }
        record["merge_key"] = _build_merge_key(record)
        if _record_is_valid(record):
            out.append(record)
    return out


def extract_financial_enrichment(
    *,
    path: str,
    full_text: str,
    chunks: Sequence[Dict[str, Any]],
    doc_type: Optional[str],
    checksum: str,
    document_id: str,
    enable_llm_fallback: bool = True,
) -> FinancialExtractionResult:
    """
    Main entry point for financial enrichment of a single document.

    Called by orchestrator.ingest_one() after doc classification and chunking.
    Returns a FinancialExtractionResult containing:
    - document_metadata: flat dict merged into every chunk and the full-text doc
      in OpenSearch. Key fields: is_financial_document, transaction_dates,
      financial_record_type, financial_metadata_source.
    - records: list of individual transaction dicts upserted to the
      financial_records OpenSearch index via financial_records_store.py.
    - source_family: the resolved family label (e.g. "tax_document", "publication")
    - used_llm_fallback: True if the LLM was called to fill extraction gaps.

    Suppression guard (FIX for misclassification bug)
    -------------------------------------------------
    If _source_family() resolves to a suppressed family (book, cv, research paper,
    etc.) we should return early with is_financial_document=False before running
    any extraction. Without this guard, a research paper containing the word
    "expense" can pass through and get labelled as a tax document.
    Recommended addition at the top of this function:

        if source_family in _SUPPRESSED_FAMILIES:
            return FinancialExtractionResult(
                document_metadata={"is_financial_document": False},
                records=[],
                source_family=source_family,
                used_llm_fallback=False,
            )
    """
    source_family = _source_family(path, doc_type, full_text)

    # SUPPRESSION GUARD: never run financial extraction on non-financial document families.
    # This prevents research papers, CVs, and course materials from being misclassified
    # as tax documents due to incidental keyword matches (e.g. "cost", "expense").
    if source_family in _SUPPRESSED_FAMILIES:
        return FinancialExtractionResult(
            document_metadata={"is_financial_document": False},
            records=[],
            source_family=source_family,
            used_llm_fallback=False,
        )

    all_dates = _extract_dates(full_text)
    all_years = _extract_years(full_text)
    amount_matches = _extract_amount_matches(full_text)
    summary_amounts = _limited_unique(
        [item["amount"] for item in amount_matches if isinstance(item.get("amount"), float)],
        limit=25,
    )
    summary_counterparties = _limited_unique(
        [value for value in [_extract_counterparty(full_text)] if value],
        limit=20,
    )
    tax_signals = _extract_tax_signals(full_text)
    expense_category = _derive_expense_category(full_text)

    records: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        chunk_text = str(chunk.get("text") or "")
        if not chunk_text.strip():
            continue
        chunk_dates = _extract_dates(chunk_text)
        chunk_amounts = _extract_amount_matches(chunk_text)
        if not chunk_amounts:
            continue
        chunk_counterparty = _extract_counterparty(chunk_text)
        chunk_id = chunk.get("id")
        if not isinstance(chunk_id, str) or not chunk_id:
            chunk_id = f"{checksum}:{idx}"

        for amount_entry in chunk_amounts:
            date_value = chunk_dates[0] if chunk_dates else (all_dates[0] if all_dates else None)
            if not date_value:
                continue
            start = int(amount_entry.get("start") or 0)
            end = int(amount_entry.get("end") or start)
            span_start = max(0, start - 60)
            span_end = min(len(chunk_text), end + 60)
            span_text = chunk_text[span_start:span_end].strip()
            record = {
                "record_type": _default_record_type(source_family, chunk_text),
                "date": date_value,
                "amount": float(amount_entry.get("amount") or 0.0),
                "currency": _normalize_currency(str(amount_entry.get("currency") or "")),
                "counterparty": chunk_counterparty,
                "description": chunk_text[:220].strip(),
                "confidence": _record_confidence(
                    has_date=True,
                    has_counterparty=bool(chunk_counterparty),
                    llm=False,
                ),
                "document_id": document_id,
                "checksum": checksum,
                "chunk_id": chunk_id,
                "extraction_method": "deterministic",
                "source_text_span": span_text,
                "financial_record_version": "v1",
                "source_family": source_family,
                "year": int(date_value[:4]),
                "source_links": [
                    {
                        "document_id": document_id,
                        "checksum": checksum,
                        "chunk_id": chunk_id,
                        "source_text_span": span_text,
                        "extraction_method": "deterministic",
                        "confidence": _record_confidence(
                            has_date=True,
                            has_counterparty=bool(chunk_counterparty),
                            llm=False,
                        ),
                    }
                ],
            }
            record["merge_key"] = _build_merge_key(record)
            if _record_is_valid(record):
                records.append(record)
                if chunk_counterparty and chunk_counterparty not in summary_counterparties:
                    summary_counterparties.append(chunk_counterparty)

    records = merge_duplicate_records(records)
    transaction_dates = _sorted_unique([str(r.get("date") or "") for r in records])
    tax_years = sorted(
        {
            int(year)
            for year in all_years
            if re.search(rf"\b{year}\b.{0,20}\b(tax|steuer|vat)\b|\b(tax|steuer|vat)\b.{0,20}\b{year}\b", full_text, re.IGNORECASE)
        }
    )

    is_financial_document = bool(
        records
        or source_family in _PREFERRED_FINANCIAL_FAMILIES
        or tax_signals
        or (len(summary_amounts) >= 2 and bool(all_dates))
    )

    confident_family = source_family in (_PREFERRED_FINANCIAL_FAMILIES | {"official_letter"})
    has_date_and_amount = bool(transaction_dates) and bool(summary_amounts)
    likely_financial = is_financial_document or source_family in _PREFERRED_FINANCIAL_FAMILIES
    counterparty_empty_for_likely_financial = likely_financial and bool(summary_amounts) and not summary_counterparties
    needs_llm = likely_financial and (
        not (has_date_and_amount and confident_family)
        or counterparty_empty_for_likely_financial
        or source_family == "archive_misc"
    )

    used_llm_fallback = False
    fallback_reason: Optional[str] = None
    if needs_llm and enable_llm_fallback:
        fallback_reason = "missing_key_fields_or_ambiguous_family"
        llm_payload = _try_llm_fallback(path, full_text, source_family)
        llm_metadata = llm_payload.get("document_metadata", {}) if isinstance(llm_payload, dict) else {}
        llm_records_raw = llm_payload.get("records", []) if isinstance(llm_payload, dict) else []
        llm_records = []
        if isinstance(llm_records_raw, list):
            llm_records = _normalize_llm_records(
                [item for item in llm_records_raw if isinstance(item, dict)],
                document_id=document_id,
                checksum=checksum,
                source_family=source_family,
            )
        if llm_records or llm_metadata:
            used_llm_fallback = True
            records = merge_duplicate_records([*records, *llm_records])
            if not transaction_dates:
                transaction_dates = _sorted_unique(
                    [str(r.get("date") or "") for r in llm_records if r.get("date")]
                )
            if not summary_amounts and isinstance(llm_metadata, dict):
                summary_amounts = _limited_unique(
                    [
                        _normalize_amount(str(amount))
                        for amount in (llm_metadata.get("amounts") or [])
                        if _normalize_amount(str(amount)) is not None
                    ],
                    limit=25,
                )
            if not summary_counterparties and isinstance(llm_metadata, dict):
                summary_counterparties = _limited_unique(
                    [
                        str(name).strip()
                        for name in (llm_metadata.get("counterparties") or [])
                        if str(name).strip()
                    ],
                    limit=20,
                )
            if not all_dates and isinstance(llm_metadata, dict):
                all_dates = _sorted_unique(
                    [
                        _parse_date_token(str(value)) or ""
                        for value in (llm_metadata.get("transaction_dates") or [])
                    ]
                )
            if not tax_years and isinstance(llm_metadata, dict):
                tax_years = sorted(
                    {
                        int(value)
                        for value in (llm_metadata.get("tax_years_referenced") or [])
                        if isinstance(value, int)
                    }
                )
            if not expense_category and isinstance(llm_metadata, dict):
                value = str(llm_metadata.get("expense_category") or "").strip()
                expense_category = value or None

    document_date = all_dates[0] if all_dates else (transaction_dates[0] if transaction_dates else None)
    metadata = {
        "is_financial_document": is_financial_document,
        "document_date": document_date,
        "mentioned_years": all_years,
        "transaction_dates": transaction_dates,
        "tax_years_referenced": tax_years,
        "amounts": summary_amounts,
        "counterparties": summary_counterparties,
        "tax_relevance_signals": tax_signals,
        "expense_category": expense_category,
        "financial_record_type": _default_record_type(source_family, full_text),
        "financial_metadata_version": "v1",
        "financial_metadata_source": "hybrid" if used_llm_fallback else "deterministic",
    }

    return FinancialExtractionResult(
        document_metadata=metadata,
        records=records,
        source_family=source_family,
        used_llm_fallback=used_llm_fallback,
        fallback_reason=fallback_reason if used_llm_fallback else None,
    )


def record_checksum(record: Dict[str, Any]) -> str:
    merge_key = str(record.get("merge_key") or _build_merge_key(record))
    return sha1(merge_key.encode("utf-8")).hexdigest()
