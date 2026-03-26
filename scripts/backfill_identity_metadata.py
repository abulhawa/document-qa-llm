#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on sys.path when run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CHUNKS_INDEX, FULLTEXT_INDEX, logger  # noqa: E402
from core.opensearch_client import get_client  # noqa: E402
from ingestion.doc_classifier import classify_document  # noqa: E402
from utils.opensearch_utils import ensure_identity_metadata_mappings  # noqa: E402


_IDENTITY_FIELDS = ("doc_type", "person_name", "authority_rank")


def _non_null_identity_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for field in _IDENTITY_FIELDS:
        value = metadata.get(field)
        if value is not None:
            cleaned[field] = value
    return cleaned


def _build_fulltext_patch(
    source: Dict[str, Any],
    classified: Dict[str, Any],
    *,
    overwrite: bool,
) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    for field, value in _non_null_identity_metadata(classified).items():
        if overwrite:
            patch[field] = value
            continue
        existing = source.get(field)
        if existing is None:
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

    for field, value in _non_null_identity_metadata(metadata).items():
        script_params[field] = value
        if overwrite:
            script_parts.append(f"ctx._source.{field} = params.{field};")
        else:
            script_parts.append(
                f"if (!ctx._source.containsKey('{field}') || ctx._source.{field} == null) "
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


def backfill_identity_metadata(
    *,
    batch_size: int = 200,
    limit: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    ensure_mappings: bool = True,
) -> Dict[str, int]:
    if ensure_mappings:
        ensure_identity_metadata_mappings()

    client = get_client()
    stats: Dict[str, int] = {
        "scanned_fulltext_docs": 0,
        "classified_docs": 0,
        "skipped_no_identity_metadata": 0,
        "skipped_unchanged_fulltext": 0,
        "fulltext_updates": 0,
        "fulltext_would_update": 0,
        "chunk_update_calls": 0,
        "chunk_would_update_calls": 0,
        "chunk_docs_updated": 0,
        "errors": 0,
    }

    processed = 0
    scroll_id = ""

    try:
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
                    "person_name",
                    "authority_rank",
                ],
                "sort": [{"_id": "asc"}],
            },
            params={"scroll": "2m"},
        )
        scroll_id = response.get("_scroll_id", "")

        while True:
            hits = response.get("hits", {}).get("hits", []) or []
            if not hits:
                break

            for hit in hits:
                if limit is not None and processed >= limit:
                    return stats

                processed += 1
                stats["scanned_fulltext_docs"] += 1

                source = hit.get("_source") or {}
                doc_id = hit.get("_id")
                checksum = source.get("checksum")
                path = str(source.get("path") or "")
                filetype = str(source.get("filetype") or "")
                text_full = str(source.get("text_full") or "")

                if not doc_id or not checksum:
                    stats["errors"] += 1
                    continue

                classified = classify_document(path, filetype, text_full)
                identity_metadata = _non_null_identity_metadata(classified)
                if not identity_metadata:
                    stats["skipped_no_identity_metadata"] += 1
                    continue

                stats["classified_docs"] += 1
                fulltext_patch = _build_fulltext_patch(
                    source,
                    classified,
                    overwrite=overwrite,
                )
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
                            logger.error(
                                "Failed to update fulltext identity metadata for id=%s: %s",
                                doc_id,
                                exc,
                            )
                            stats["errors"] += 1
                            continue
                else:
                    stats["skipped_unchanged_fulltext"] += 1

                chunk_update = _chunk_update_body(
                    str(checksum),
                    classified,
                    overwrite=overwrite,
                )
                if not chunk_update.get("script", {}).get("source"):
                    continue

                if dry_run:
                    stats["chunk_would_update_calls"] += 1
                    continue

                try:
                    update_resp = client.update_by_query(
                        index=CHUNKS_INDEX,
                        body=chunk_update,
                        params={"refresh": "false", "conflicts": "proceed"},
                    )
                    stats["chunk_update_calls"] += 1
                    stats["chunk_docs_updated"] += int(update_resp.get("updated", 0))
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed chunk metadata backfill for checksum=%s: %s",
                        checksum,
                        exc,
                    )
                    stats["errors"] += 1

            if not scroll_id:
                break
            response = client.scroll(scroll_id=scroll_id, params={"scroll": "2m"})
            scroll_id = response.get("_scroll_id", scroll_id)
    finally:
        if scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception:  # noqa: BLE001
                pass

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-off backfill for identity metadata on existing indexed documents."
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing identity metadata instead of only filling missing fields.",
    )
    parser.add_argument(
        "--skip-ensure-mappings",
        action="store_true",
        help="Skip non-destructive mapping checks/additions before running backfill.",
    )
    args = parser.parse_args()

    stats = backfill_identity_metadata(
        batch_size=max(1, args.batch_size),
        limit=args.limit,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        ensure_mappings=not args.skip_ensure_mappings,
    )
    print(
        "Backfill complete: "
        "scanned_fulltext_docs={scanned_fulltext_docs} "
        "classified_docs={classified_docs} "
        "skipped_no_identity_metadata={skipped_no_identity_metadata} "
        "skipped_unchanged_fulltext={skipped_unchanged_fulltext} "
        "fulltext_updates={fulltext_updates} "
        "fulltext_would_update={fulltext_would_update} "
        "chunk_update_calls={chunk_update_calls} "
        "chunk_would_update_calls={chunk_would_update_calls} "
        "chunk_docs_updated={chunk_docs_updated} "
        "errors={errors}".format(**stats)
    )


if __name__ == "__main__":
    main()
