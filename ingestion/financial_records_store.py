from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from opensearchpy import exceptions

from config import FINANCIAL_RECORDS_INDEX, logger
from core.opensearch_client import get_client
from ingestion.financial_extractor import record_checksum
from utils.opensearch_utils import ensure_financial_records_index


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge_source_links(
    existing_links: Sequence[Dict[str, Any]],
    new_links: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    seen = set()
    merged: List[Dict[str, Any]] = []
    for link in [*existing_links, *new_links]:
        if not isinstance(link, dict):
            continue
        key = (
            str(link.get("document_id") or ""),
            str(link.get("checksum") or ""),
            str(link.get("chunk_id") or ""),
            str(link.get("source_text_span") or ""),
            str(link.get("extraction_method") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(link))
    return merged[:50]


def _merge_records(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    merged_links = _merge_source_links(
        list(existing.get("source_links") or []),
        list(incoming.get("source_links") or []),
    )
    merged["source_links"] = merged_links
    merged["source_count"] = len(merged_links)
    merged["confidence"] = max(
        _coerce_float(existing.get("confidence"), default=0.0),
        _coerce_float(incoming.get("confidence"), default=0.0),
    )
    existing_method = str(existing.get("extraction_method") or "").strip()
    incoming_method = str(incoming.get("extraction_method") or "").strip()
    if existing_method and incoming_method and existing_method != incoming_method:
        merged["extraction_method"] = "hybrid"
    elif incoming_method:
        merged["extraction_method"] = incoming_method
    return merged


def upsert_financial_records(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    ensure_financial_records_index()
    client = get_client()
    stats = {
        "processed": 0,
        "created": 0,
        "updated": 0,
        "errors": 0,
    }
    for raw in records:
        if not isinstance(raw, dict):
            continue
        record = dict(raw)
        record_id = record_checksum(record)
        stats["processed"] += 1
        try:
            existing = None
            try:
                existing_resp = client.get(index=FINANCIAL_RECORDS_INDEX, id=record_id)
                existing = existing_resp.get("_source") or {}
            except exceptions.NotFoundError:
                existing = None

            payload = record if existing is None else _merge_records(existing, record)
            client.index(
                index=FINANCIAL_RECORDS_INDEX,
                id=record_id,
                body=payload,
                op_type="index",  # pyright: ignore[reportCallIssue]
                refresh=False,  # pyright: ignore[reportCallIssue]
            )
            if existing is None:
                stats["created"] += 1
            else:
                stats["updated"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to upsert financial record id=%s: %s",
                record_id,
                exc,
            )
            stats["errors"] += 1
    return stats


def fetch_financial_records(
    *,
    checksums: Sequence[str],
    year: Optional[int] = None,
    size: int = 200,
) -> List[Dict[str, Any]]:
    if not checksums:
        return []
    ensure_financial_records_index()
    client = get_client()
    checksum_values = [value for value in checksums if value]
    filters: List[Dict[str, Any]] = [
        {
            "bool": {
                "should": [
                    {"terms": {"checksum": checksum_values}},
                    {
                        "nested": {
                            "path": "source_links",
                            "query": {"terms": {"source_links.checksum": checksum_values}},
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }
    ]
    if year is not None:
        filters.append({"term": {"year": {"value": int(year)}}})

    try:
        response = client.search(
            index=FINANCIAL_RECORDS_INDEX,
            body={
                "size": max(1, int(size)),
                "query": {"bool": {"filter": filters}},
                "sort": [
                    {"confidence": {"order": "desc"}},
                    {"date": {"order": "desc", "missing": "_last"}},
                ],
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Financial sidecar search failed: %s", exc)
        return []

    hits = response.get("hits", {}).get("hits", []) or []
    return [dict(hit.get("_source") or {}) for hit in hits]
