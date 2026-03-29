from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from core.financial_records import get_financial_records_for_checksums
from qa_pipeline.types import RetrievalResult


_DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{1,2}-\d{1,2}\b|\b(19|20)\d{2}\b")
_AMOUNT_RE = re.compile(
    r"(?P<currency_1>EUR|USD|GBP|CHF)\s*(?P<amount_1>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)|"
    r"(?P<amount_2>\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*(?P<currency_2>EUR|USD|GBP|CHF)",
    re.IGNORECASE,
)
_STRONG_FALLBACK_SOURCE_FAMILIES = {
    "invoice",
    "receipt",
    "bank_statement",
    "payment_confirmation",
    "tax_document",
    "school_fee_letter",
}
_FINANCIAL_RECORD_TERMS = {
    "expense",
    "invoice",
    "receipt",
    "payment",
    "transfer",
    "refund",
    "tax_payment",
}


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


def _year_from_token(token: str) -> Optional[int]:
    text = str(token or "").strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _safe_currency(raw: Any) -> str:
    value = str(raw or "").strip().upper()
    if not value or value == "UNKNOWN":
        return ""
    return value


def _clean_source_label(raw_source: Any) -> str:
    source = str(raw_source or "").strip()
    if not source:
        return "unknown source"
    normalized = source.replace("\\", "/")
    if "/" in normalized:
        filename = os.path.basename(normalized).strip()
        if filename:
            return filename
    return source


def _format_amount(amount: Any, currency: Any) -> str:
    if isinstance(amount, (int, float)):
        base = f"{float(amount):.2f}"
        cur = _safe_currency(currency)
        return f"{base} {cur}".strip()
    return "not normalized"


def _extract_first_amount(text: str) -> tuple[Optional[float], str]:
    match = _AMOUNT_RE.search(text or "")
    if not match:
        return None, ""
    raw_amount = match.group("amount_1") or match.group("amount_2") or ""
    amount = _normalize_amount(raw_amount)
    if amount is None:
        return None, ""
    raw_currency = match.group("currency_1") or match.group("currency_2") or ""
    return amount, _safe_currency(raw_currency)


def _doc_candidate_years(doc: Any) -> set[int]:
    years: set[int] = set()
    for raw in (doc.mentioned_years or []):
        try:
            years.add(int(raw))
        except (TypeError, ValueError):
            continue
    for raw in (doc.tax_years_referenced or []):
        try:
            years.add(int(raw))
        except (TypeError, ValueError):
            continue
    for raw_date in (doc.transaction_dates or []):
        year = _year_from_token(str(raw_date))
        if year is not None:
            years.add(year)
    doc_year = _year_from_token(str(doc.document_date or ""))
    if doc_year is not None:
        years.add(doc_year)
    return years


def _doc_matches_target_year(doc: Any, target_year: Optional[int], text: str) -> bool:
    if target_year is None:
        return True
    doc_years = _doc_candidate_years(doc)
    if doc_years:
        return target_year in doc_years

    text_years = {
        year
        for year in (_year_from_token(match.group(0)) for match in _DATE_RE.finditer(text or ""))
        if year is not None
    }
    if text_years:
        return target_year in text_years
    return True


def _doc_has_structured_financial_signal(doc: Any) -> bool:
    if bool(doc.is_financial_document):
        return True
    source_family = str(doc.source_family or "").strip().lower()
    if source_family in _STRONG_FALLBACK_SOURCE_FAMILIES:
        return True
    record_type = str(doc.financial_record_type or "").strip().lower()
    if any(term in record_type for term in _FINANCIAL_RECORD_TERMS):
        return True
    if doc.transaction_dates:
        return True
    return False


def _best_date_token(text: str, target_year: Optional[int]) -> str:
    first_token = ""
    for match in _DATE_RE.finditer(text or ""):
        token = str(match.group(0) or "").strip()
        if not token:
            continue
        if not first_token:
            first_token = token
        token_year = _year_from_token(token)
        if target_year is None or token_year == target_year:
            return token
    return first_token


def _parse_year_from_record(record: Dict[str, Any]) -> Optional[int]:
    year_value = record.get("year")
    if isinstance(year_value, int):
        return year_value
    date_text = str(record.get("date") or "").strip()
    if len(date_text) >= 4 and date_text[:4].isdigit():
        return int(date_text[:4])
    return None


def _format_item(item: Dict[str, Any]) -> str:
    date = item.get("date") or "unknown"
    amount_text = _format_amount(item.get("amount"), item.get("currency"))
    counterparty = item.get("counterparty") or "unknown"
    source = _clean_source_label(item.get("source"))
    line = (
        f"- Date: {date} | Amount: {amount_text} | "
        f"Counterparty: {counterparty} | Source: {source}"
    )
    note = str(item.get("note") or "").strip()
    if note:
        return f"{line} | Note: {note}"
    return line


def _concept_match(record: Dict[str, Any], target_concept: Optional[str]) -> bool:
    if not target_concept:
        return True
    record_type = str(record.get("record_type") or "").lower()
    if target_concept == "expenses":
        return "expense" in record_type or "invoice" in record_type or "receipt" in record_type
    if target_concept == "payments":
        return "payment" in record_type or "transfer" in record_type
    return True


def _source_from_record(
    record: Dict[str, Any],
    fallback_source: str,
    *,
    checksum_to_source: Dict[str, str],
) -> str:
    links = record.get("source_links")
    if isinstance(links, list) and links:
        first = links[0]
        if isinstance(first, dict):
            checksum = str(first.get("checksum") or "").strip()
            chunk_id = str(first.get("chunk_id") or "").strip()
            linked_source = checksum_to_source.get(checksum)
            if linked_source:
                if chunk_id:
                    return f"{linked_source} (chunk evidence)"
                return linked_source
            if checksum and chunk_id:
                return f"document {checksum[:10]} (chunk evidence)"
            if checksum:
                return f"document {checksum[:10]}"
    fallback = checksum_to_source.get(fallback_source) or fallback_source
    return _clean_source_label(fallback)


def _bucket_subject(target_concept: Optional[str]) -> str:
    if target_concept == "payments":
        return "payments"
    return "expenses"


def _fallback_items_from_docs(
    retrieval: RetrievalResult,
    target_year: Optional[int],
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    strong_items: List[Dict[str, Any]] = []
    weak_items: List[Dict[str, Any]] = []
    skipped_year_mismatch = 0
    skipped_no_amount = 0
    seen: set[tuple[str, str, float, str]] = set()

    for doc in retrieval.documents:
        text = str(doc.text or "")
        if not _doc_matches_target_year(doc, target_year, text):
            skipped_year_mismatch += 1
            continue
        amount, currency = _extract_first_amount(text)
        if amount is None:
            skipped_no_amount += 1
            continue
        date_token = _best_date_token(text, target_year) or (
            str(target_year) if target_year is not None else "unknown"
        )
        source = _clean_source_label(doc.path)
        dedupe_key = (source, date_token, amount, currency)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        item = {
            "date": date_token,
            "amount": amount,
            "currency": currency,
            "counterparty": "unknown",
            "source": source,
            "note": "mentioned in document text but no normalized payment confirmation",
        }
        if _doc_has_structured_financial_signal(doc):
            strong_items.append(item)
        else:
            weak = dict(item)
            weak["note"] = "weak mention-only signal; no normalized financial metadata"
            weak_items.append(weak)

    if strong_items:
        return strong_items, {
            "strong_items": len(strong_items),
            "weak_retained": 0,
            "weak_suppressed": len(weak_items),
            "skipped_no_amount": skipped_no_amount,
            "skipped_year_mismatch": skipped_year_mismatch,
        }

    retained_weak = weak_items[:2]
    return retained_weak, {
        "strong_items": 0,
        "weak_retained": len(retained_weak),
        "weak_suppressed": max(0, len(weak_items) - len(retained_weak)),
        "skipped_no_amount": skipped_no_amount,
        "skipped_year_mismatch": skipped_year_mismatch,
    }


def build_financial_answer(
    *,
    retrieval: RetrievalResult,
    target_year: Optional[int],
    target_entity: Optional[str],
    target_concept: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    checksum_to_source: Dict[str, str] = {}
    for doc in retrieval.documents:
        checksum = str(doc.checksum or "").strip()
        if checksum and checksum not in checksum_to_source:
            checksum_to_source[checksum] = _clean_source_label(doc.path)

    checksums = [str(doc.checksum or "").strip() for doc in retrieval.documents if doc.checksum]
    unique_checksums = [value for value in dict.fromkeys(checksums) if value]
    sidecar_records = get_financial_records_for_checksums(
        checksums=unique_checksums,
        year=target_year,
        size=300,
    )

    clearly_supported: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    mentioned_not_confirmed: List[Dict[str, Any]] = []

    fallback_metrics = {
        "strong_items": 0,
        "weak_retained": 0,
        "weak_suppressed": 0,
        "skipped_no_amount": 0,
        "skipped_year_mismatch": 0,
    }

    for record in sidecar_records:
        if not _concept_match(record, target_concept):
            continue
        confidence = float(record.get("confidence") or 0.0)
        year = _parse_year_from_record(record)
        source = _source_from_record(
            record,
            str(record.get("checksum") or "unknown"),
            checksum_to_source=checksum_to_source,
        )
        base_item = {
            "date": record.get("date"),
            "amount": record.get("amount"),
            "currency": record.get("currency"),
            "counterparty": record.get("counterparty"),
            "source": source,
            "note": "",
        }
        if target_year is not None and year is not None and year != target_year:
            base_item["note"] = f"recorded year is {year}, not {target_year}"
            mentioned_not_confirmed.append(base_item)
            continue
        if target_year is not None and year is None:
            base_item["note"] = "record has no normalized year"
            ambiguous.append(base_item)
            continue

        has_evidence = bool(record.get("source_links"))
        if has_evidence and confidence >= 0.65:
            clearly_supported.append(base_item)
        else:
            if confidence < 0.65:
                base_item["note"] = "low extraction confidence"
            else:
                base_item["note"] = "no direct evidence link"
            ambiguous.append(base_item)

    expected_fin_docs = {
        str(doc.checksum or "").strip()
        for doc in retrieval.documents
        if doc.is_financial_document and doc.checksum
    }
    sidecar_covered = {
        str(record.get("checksum") or "").strip()
        for record in sidecar_records
        if str(record.get("checksum") or "").strip()
    }
    normalized_coverage_incomplete = bool(expected_fin_docs - sidecar_covered)

    if not sidecar_records or normalized_coverage_incomplete:
        fallback_items, fallback_metrics = _fallback_items_from_docs(
            retrieval,
            target_year=target_year,
        )
        mentioned_not_confirmed.extend(fallback_items)

    if not clearly_supported and not ambiguous and not mentioned_not_confirmed:
        return "I don't know.", {
            "financial_query_mode": True,
            "target_entity": target_entity,
            "target_year": target_year,
            "target_concept": target_concept,
            "fallback_retrieval_used": bool(retrieval.stage_metadata.get("fallback_used")),
            "normalized_record_coverage_incomplete": normalized_coverage_incomplete,
            "sidecar_records_found": len(sidecar_records),
            "fallback_item_metrics": fallback_metrics,
        }

    header_year = str(target_year) if target_year is not None else "unspecified year"
    bucket_subject = _bucket_subject(target_concept)
    year_bucket_prefix = f"{target_year} " if target_year is not None else ""
    lines: List[str] = [
        f"Financial evidence summary for {header_year}:",
    ]
    if target_entity:
        lines.append(f"Target entity hint: {target_entity} (metadata only; not a hard retrieval filter)")
    if retrieval.stage_metadata.get("fallback_used"):
        lines.append(
            "Retrieval disclosure: fallback retrieval stages were used to fill residual candidate budget."
        )
    if normalized_coverage_incomplete:
        lines.append(
            "Coverage disclosure: normalized financial-record coverage is incomplete; fallback chunk evidence is included."
        )
    if fallback_metrics.get("weak_suppressed", 0) > 0:
        lines.append(
            "Evidence disclosure: weak mention-only candidates were suppressed when stronger financial candidates were available."
        )

    lines.append("")
    lines.append(f"Clearly supported {year_bucket_prefix}{bucket_subject}:")
    if clearly_supported:
        lines.extend(_format_item(item) for item in clearly_supported)
    else:
        lines.append("- None")

    lines.append("")
    lines.append(f"Possible but ambiguous {year_bucket_prefix}{bucket_subject}:")
    if ambiguous:
        lines.extend(_format_item(item) for item in ambiguous)
    else:
        lines.append("- None")

    lines.append("")
    if target_year is not None:
        lines.append(f"Mentioned items not confirmed as paid in {target_year}:")
    else:
        lines.append("Mentioned items not confirmed as paid in target year:")
    if mentioned_not_confirmed:
        lines.extend(_format_item(item) for item in mentioned_not_confirmed)
    else:
        lines.append("- None")

    lines.append("")
    lines.append(
        "Tax relevance note: items are evidence-linked only; tax deductibility is tentative unless directly supported."
    )
    return "\n".join(lines), {
        "financial_query_mode": True,
        "target_entity": target_entity,
        "target_year": target_year,
        "target_concept": target_concept,
        "fallback_retrieval_used": bool(retrieval.stage_metadata.get("fallback_used")),
        "normalized_record_coverage_incomplete": normalized_coverage_incomplete,
        "sidecar_records_found": len(sidecar_records),
        "fallback_item_metrics": fallback_metrics,
        "bucket_counts": {
            "clearly_supported": len(clearly_supported),
            "ambiguous": len(ambiguous),
            "mentioned_not_confirmed": len(mentioned_not_confirmed),
        },
    }
