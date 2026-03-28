from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from core.financial_records import get_financial_records_for_checksums
from qa_pipeline.types import RetrievalResult


_DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{1,2}-\d{1,2}\b|\b(19|20)\d{2}\b")
_AMOUNT_RE = re.compile(
    r"\b(?:EUR|USD|GBP|CHF)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\b|"
    r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?\s*(?:EUR|USD|GBP|CHF)\b",
    re.IGNORECASE,
)


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
    amount = item.get("amount")
    amount_text = f"{float(amount):.2f}" if isinstance(amount, (int, float)) else "unknown"
    currency = item.get("currency") or "unknown"
    counterparty = item.get("counterparty") or "unknown"
    source = item.get("source") or "unknown"
    line = (
        f"- Date: {date} | Amount: {amount_text} {currency} | "
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


def _source_from_record(record: Dict[str, Any], fallback_source: str) -> str:
    links = record.get("source_links")
    if isinstance(links, list) and links:
        first = links[0]
        if isinstance(first, dict):
            checksum = str(first.get("checksum") or "").strip()
            chunk_id = str(first.get("chunk_id") or "").strip()
            if checksum and chunk_id:
                return f"{checksum}#{chunk_id}"
            if checksum:
                return checksum
    return fallback_source


def _bucket_subject(target_concept: Optional[str]) -> str:
    if target_concept == "payments":
        return "payments"
    return "expenses"


def _fallback_items_from_docs(
    retrieval: RetrievalResult,
    target_year: Optional[int],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for doc in retrieval.documents:
        text = str(doc.text or "")
        year_match = None
        for match in _DATE_RE.finditer(text):
            token = match.group(0)
            if token and token[:4].isdigit():
                year = int(token[:4])
                if target_year is None or year == target_year:
                    year_match = token
                    break
        amount_match = _AMOUNT_RE.search(text)
        if year_match is None and amount_match is None:
            continue
        items.append(
            {
                "date": year_match or "unknown",
                "amount": None,
                "currency": "unknown",
                "counterparty": "unknown",
                "source": doc.path,
                "note": "mentioned in document text but no normalized payment confirmation",
            }
        )
    return items


def build_financial_answer(
    *,
    retrieval: RetrievalResult,
    target_year: Optional[int],
    target_entity: Optional[str],
    target_concept: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
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

    for record in sidecar_records:
        if not _concept_match(record, target_concept):
            continue
        confidence = float(record.get("confidence") or 0.0)
        year = _parse_year_from_record(record)
        source = _source_from_record(record, str(record.get("checksum") or "unknown"))
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
        mentioned_not_confirmed.extend(
            _fallback_items_from_docs(retrieval, target_year=target_year)
        )

    if not clearly_supported and not ambiguous and not mentioned_not_confirmed:
        return "I don't know.", {
            "financial_query_mode": True,
            "target_entity": target_entity,
            "target_year": target_year,
            "target_concept": target_concept,
            "fallback_retrieval_used": bool(retrieval.stage_metadata.get("fallback_used")),
            "normalized_record_coverage_incomplete": normalized_coverage_incomplete,
            "sidecar_records_found": len(sidecar_records),
        }

    header_year = str(target_year) if target_year is not None else "unspecified year"
    bucket_subject = _bucket_subject(target_concept)
    year_bucket_prefix = f"{target_year} " if target_year is not None else ""
    lines: List[str] = [
        f"Financial evidence summary for {header_year}:",
    ]
    if target_entity:
        lines.append(f"Target entity: {target_entity}")
    if retrieval.stage_metadata.get("fallback_used"):
        lines.append(
            "Retrieval disclosure: fallback retrieval stages were used to fill residual candidate budget."
        )
    if normalized_coverage_incomplete:
        lines.append(
            "Coverage disclosure: normalized financial-record coverage is incomplete; fallback chunk evidence is included."
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
        "bucket_counts": {
            "clearly_supported": len(clearly_supported),
            "ambiguous": len(ambiguous),
            "mentioned_not_confirmed": len(mentioned_not_confirmed),
        },
    }
