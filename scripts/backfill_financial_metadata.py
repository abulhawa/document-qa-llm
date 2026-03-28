#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CHUNKS_INDEX, FULLTEXT_INDEX, QDRANT_COLLECTION, logger  # noqa: E402
from core.opensearch_client import get_client  # noqa: E402
from ingestion.financial_extractor import extract_financial_enrichment  # noqa: E402
from ingestion.financial_records_store import upsert_financial_records  # noqa: E402
from utils.opensearch_utils import (  # noqa: E402
    ensure_financial_metadata_mappings,
    ensure_financial_records_index,
)


_FINANCIAL_FIELDS = (
    "is_financial_document",
    "document_date",
    "mentioned_years",
    "transaction_dates",
    "tax_years_referenced",
    "amounts",
    "counterparties",
    "tax_relevance_signals",
    "expense_category",
    "financial_record_type",
    "financial_metadata_version",
    "financial_metadata_source",
)


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_filter_values(raw_values: Optional[Sequence[str]]) -> set[str] | None:
    if not raw_values:
        return None
    normalized: set[str] = set()
    for item in raw_values:
        for piece in str(item).split(","):
            token = _normalize_token(piece)
            if token:
                normalized.add(token)
    return normalized or None


def _build_fulltext_patch(
    source: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    overwrite: bool,
) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    for field in _FINANCIAL_FIELDS:
        value = metadata.get(field)
        if value is None:
            continue
        if overwrite:
            patch[field] = value
            continue
        existing = source.get(field)
        if existing is None or (isinstance(existing, list) and not existing):
            patch[field] = value
    return patch


def _chunk_update_body(
    checksum: str,
    metadata: Dict[str, Any],
    *,
    overwrite: bool,
) -> Dict[str, Any]:
    script_parts: list[str] = []
    script_params: Dict[str, Any] = {}
    for field in _FINANCIAL_FIELDS:
        value = metadata.get(field)
        if value is None:
            continue
        script_params[field] = value
        if overwrite:
            script_parts.append(f"ctx._source.{field} = params.{field};")
        else:
            script_parts.append(
                f"if (!ctx._source.containsKey('{field}') || ctx._source.{field} == null || "
                f"(ctx._source.{field} instanceof List && ctx._source.{field}.isEmpty())) "
                f"{{ ctx._source.{field} = params.{field}; }}"
            )
    return {
        "query": {"term": {"checksum": {"value": checksum}}},
        "script": {
            "lang": "painless",
            "source": " ".join(script_parts),
            "params": script_params,
        },
    }


def _iter_fulltext_hits(
    client,
    *,
    batch_size: int,
    limit: int | None,
    skip: int = 0,
    scroll_keepalive: str = "30m",
) -> Iterable[Dict[str, Any]]:
    response = client.search(
        index=FULLTEXT_INDEX,
        body={
            "size": batch_size,
            "query": {"match_all": {}},
            "_source": [
                "path",
                "filetype",
                "checksum",
                "text_full",
                "doc_type",
                "id",
                *_FINANCIAL_FIELDS,
            ],
            "sort": [{"_id": "asc"}],
        },
        params={"scroll": scroll_keepalive},
    )
    scroll_id = response.get("_scroll_id", "")
    seen = 0
    emitted = 0
    try:
        while True:
            hits = response.get("hits", {}).get("hits", []) or []
            if not hits:
                break
            for hit in hits:
                seen += 1
                if skip > 0 and seen <= skip:
                    continue
                if limit is not None and emitted >= limit:
                    return
                emitted += 1
                yield hit
            if limit is not None and emitted >= limit:
                break
            if not scroll_id:
                break
            try:
                response = client.scroll(scroll_id=scroll_id, params={"scroll": scroll_keepalive})
                scroll_id = response.get("_scroll_id", scroll_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Scroll read failed (continuing with processed subset): %s", exc)
                break
    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:  # noqa: BLE001
                pass


def _fetch_chunks_for_checksum(client, checksum: str, *, batch_size: int = 200) -> list[Dict[str, Any]]:
    after = None
    chunks: list[Dict[str, Any]] = []
    while True:
        body: Dict[str, Any] = {
            "size": batch_size,
            "query": {"term": {"checksum": {"value": checksum}}},
            "_source": ["id", "text", "chunk_index"],
            "sort": [{"chunk_index": "asc"}, {"_id": "asc"}],
        }
        if after:
            body["search_after"] = after
        response = client.search(index=CHUNKS_INDEX, body=body)
        hits = response.get("hits", {}).get("hits", []) or []
        if not hits:
            break
        for hit in hits:
            source = dict(hit.get("_source") or {})
            source["id"] = source.get("id") or hit.get("_id")
            chunks.append(source)
        after = hits[-1].get("sort")
        if not after:
            break
    return chunks


def _qdrant_payload_fields(metadata: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field in _FINANCIAL_FIELDS:
        value = metadata.get(field)
        if value is not None:
            payload[field] = value
    return payload


def _get_qdrant_components():
    from qdrant_client import models as qdrant_models
    from utils.qdrant_utils import client as configured_qdrant_client

    return configured_qdrant_client, qdrant_models


def _apply_qdrant_metadata(
    *,
    checksum: str,
    payload: Dict[str, Any],
    dry_run: bool,
) -> int:
    if not payload:
        return 0
    qdrant_client, qdrant_models = _get_qdrant_components()
    points_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="checksum",
                match=qdrant_models.MatchValue(value=checksum),
            )
        ]
    )
    if dry_run:
        try:
            count_result = qdrant_client.count(
                collection_name=QDRANT_COLLECTION,
                count_filter=points_filter,
                exact=True,
            )
            return int(getattr(count_result, "count", 0))
        except Exception:  # noqa: BLE001
            return 0

    try:
        response = qdrant_client.set_payload(
            collection_name=QDRANT_COLLECTION,
            points=points_filter,
            payload=payload,
            wait=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant payload update failed for checksum=%s: %s", checksum, exc)
        return 0

    response_any = response
    if isinstance(response_any, dict):
        return int(response_any.get("result", {}).get("count", 0))
    result = getattr(response_any, "result", None)
    if isinstance(result, dict):
        return int(result.get("count", 0))
    return 0


def backfill_financial_metadata(
    *,
    batch_size: int = 100,
    limit: int | None = None,
    skip: int = 0,
    scroll_keepalive: str = "30m",
    dry_run: bool = False,
    overwrite: bool = False,
    ensure_mappings: bool = True,
    target_doc_types: Sequence[str] | None = None,
    target_source_families: Sequence[str] | None = None,
    enable_llm_fallback: bool = False,
) -> Dict[str, int]:
    if ensure_mappings:
        ensure_financial_metadata_mappings()
        ensure_financial_records_index()

    target_doc_type_set = _normalize_filter_values(target_doc_types)
    target_source_family_set = _normalize_filter_values(target_source_families)

    client = get_client()
    stats: Dict[str, int] = {
        "scanned_fulltext_docs": 0,
        "processed_docs": 0,
        "skipped_no_fulltext_text": 0,
        "skipped_not_in_target_doc_types": 0,
        "skipped_not_in_target_source_families": 0,
        "fulltext_updates": 0,
        "fulltext_would_update": 0,
        "chunk_update_calls": 0,
        "chunk_would_update_calls": 0,
        "chunk_docs_updated": 0,
        "qdrant_payload_updated_points": 0,
        "qdrant_payload_would_update_points": 0,
        "records_extracted": 0,
        "sidecar_records_processed": 0,
        "sidecar_records_created": 0,
        "sidecar_records_updated": 0,
        "sidecar_records_errors": 0,
        "errors": 0,
    }

    for hit in _iter_fulltext_hits(
        client,
        batch_size=max(1, batch_size),
        limit=limit,
        skip=max(0, int(skip)),
        scroll_keepalive=str(scroll_keepalive or "30m"),
    ):
        source = hit.get("_source") or {}
        doc_id = hit.get("_id")
        checksum = str(source.get("checksum") or "").strip()
        path = str(source.get("path") or "")
        filetype = str(source.get("filetype") or "")
        doc_type = _normalize_token(source.get("doc_type"))
        full_text = str(source.get("text_full") or "")

        stats["scanned_fulltext_docs"] += 1
        if not doc_id or not checksum:
            stats["errors"] += 1
            continue
        if target_doc_type_set and doc_type not in target_doc_type_set:
            stats["skipped_not_in_target_doc_types"] += 1
            continue
        if not full_text.strip():
            stats["skipped_no_fulltext_text"] += 1
            continue

        try:
            chunks = _fetch_chunks_for_checksum(client, checksum)
            extraction = extract_financial_enrichment(
                path=path,
                full_text=full_text,
                chunks=chunks,
                doc_type=doc_type,
                checksum=checksum,
                document_id=str(source.get("id") or doc_id),
                enable_llm_fallback=enable_llm_fallback,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Financial backfill extraction failed for checksum=%s: %s", checksum, exc)
            stats["errors"] += 1
            continue

        source_family = _normalize_token(extraction.source_family)
        if target_source_family_set and source_family not in target_source_family_set:
            stats["skipped_not_in_target_source_families"] += 1
            continue

        metadata = extraction.document_metadata
        stats["processed_docs"] += 1
        stats["records_extracted"] += len(extraction.records)

        fulltext_patch = _build_fulltext_patch(source, metadata, overwrite=overwrite)
        if fulltext_patch:
            if dry_run:
                stats["fulltext_would_update"] += 1
            else:
                try:
                    client.update(
                        index=FULLTEXT_INDEX,
                        id=doc_id,
                        body={"doc": fulltext_patch},
                        params={"refresh": "false"},
                    )
                    stats["fulltext_updates"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Fulltext update failed for id=%s: %s", doc_id, exc)
                    stats["errors"] += 1

        chunk_update_body = _chunk_update_body(checksum, metadata, overwrite=overwrite)
        if chunk_update_body.get("script", {}).get("source"):
            if dry_run:
                stats["chunk_would_update_calls"] += 1
            else:
                try:
                    update_resp = client.update_by_query(
                        index=CHUNKS_INDEX,
                        body=chunk_update_body,
                        params={"refresh": "false", "conflicts": "proceed"},
                    )
                    stats["chunk_update_calls"] += 1
                    stats["chunk_docs_updated"] += int(update_resp.get("updated", 0))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Chunk update failed for checksum=%s: %s", checksum, exc)
                    stats["errors"] += 1

        qdrant_payload = _qdrant_payload_fields(metadata)
        if qdrant_payload:
            updated_points = _apply_qdrant_metadata(
                checksum=checksum,
                payload=qdrant_payload,
                dry_run=dry_run,
            )
            if dry_run:
                stats["qdrant_payload_would_update_points"] += int(updated_points)
            else:
                stats["qdrant_payload_updated_points"] += int(updated_points)

        if dry_run:
            continue

        if extraction.records:
            sidecar_stats = upsert_financial_records(extraction.records)
            stats["sidecar_records_processed"] += int(sidecar_stats.get("processed", 0))
            stats["sidecar_records_created"] += int(sidecar_stats.get("created", 0))
            stats["sidecar_records_updated"] += int(sidecar_stats.get("updated", 0))
            stats["sidecar_records_errors"] += int(sidecar_stats.get("errors", 0))

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill financial metadata and sidecar financial records from indexed fulltext documents."
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument(
        "--scroll-keepalive",
        type=str,
        default="30m",
        help="OpenSearch scroll keepalive window.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-ensure-mappings",
        action="store_true",
        help="Skip mapping/index ensure operations.",
    )
    parser.add_argument(
        "--target-doc-types",
        nargs="*",
        default=None,
        help="Optional doc_type filter cohort.",
    )
    parser.add_argument(
        "--target-source-families",
        nargs="*",
        default=None,
        help="Optional source-family filter cohort.",
    )
    parser.add_argument(
        "--enable-llm-fallback",
        action="store_true",
        help="Enable optional LLM fallback during extraction.",
    )
    args = parser.parse_args()

    stats = backfill_financial_metadata(
        batch_size=max(1, int(args.batch_size)),
        limit=args.limit,
        skip=max(0, int(args.skip)),
        scroll_keepalive=str(args.scroll_keepalive or "30m"),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        ensure_mappings=not args.skip_ensure_mappings,
        target_doc_types=args.target_doc_types,
        target_source_families=args.target_source_families,
        enable_llm_fallback=bool(args.enable_llm_fallback),
    )
    print(
        "Backfill complete: "
        "scanned_fulltext_docs={scanned_fulltext_docs} "
        "processed_docs={processed_docs} "
        "skipped_no_fulltext_text={skipped_no_fulltext_text} "
        "skipped_not_in_target_doc_types={skipped_not_in_target_doc_types} "
        "skipped_not_in_target_source_families={skipped_not_in_target_source_families} "
        "fulltext_updates={fulltext_updates} "
        "fulltext_would_update={fulltext_would_update} "
        "chunk_update_calls={chunk_update_calls} "
        "chunk_would_update_calls={chunk_would_update_calls} "
        "chunk_docs_updated={chunk_docs_updated} "
        "qdrant_payload_updated_points={qdrant_payload_updated_points} "
        "qdrant_payload_would_update_points={qdrant_payload_would_update_points} "
        "records_extracted={records_extracted} "
        "sidecar_records_processed={sidecar_records_processed} "
        "sidecar_records_created={sidecar_records_created} "
        "sidecar_records_updated={sidecar_records_updated} "
        "sidecar_records_errors={sidecar_records_errors} "
        "errors={errors}".format(**stats)
    )


if __name__ == "__main__":
    main()
