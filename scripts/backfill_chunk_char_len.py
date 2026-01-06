from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional

from core.opensearch_client import get_client
from opensearchpy import helpers
from qdrant_client import QdrantClient, models

from config import CHUNKS_INDEX, QDRANT_COLLECTION, QDRANT_URL, logger
from utils.opensearch_utils import ensure_chunk_char_len_mapping


def _resolve_checksum_sort_field(os_client) -> str:
    mapping = os_client.indices.get_mapping(index=CHUNKS_INDEX)
    props = (
        mapping.get(CHUNKS_INDEX, {})
        .get("mappings", {})
        .get("properties", {})
        or {}
    )
    checksum_props = props.get("checksum")
    if isinstance(checksum_props, dict):
        fields = checksum_props.get("fields", {})
        if isinstance(fields, dict) and "keyword" in fields:
            return "checksum.keyword"
    return "checksum"


def _iter_chunks(
    os_client,
    *,
    batch_size: int,
    limit: Optional[int],
    checksum_sort_field: str,
) -> Iterable[list[dict[str, Any]]]:
    after = None
    processed = 0
    while True:
        if limit is not None:
            remaining = limit - processed
            if remaining <= 0:
                break
            size = min(batch_size, remaining)
        else:
            size = batch_size

        body: Dict[str, Any] = {
            "size": size,
            "_source": ["checksum", "chunk_index", "text"],
            "sort": [
                {checksum_sort_field: "asc"},
                {"chunk_index": "asc"},
                {"_id": "asc"},
            ],
        }
        if after:
            body["search_after"] = after
        resp = os_client.search(index=CHUNKS_INDEX, body=body)
        hits = resp.get("hits", {}).get("hits", []) or []
        if not hits:
            break
        yield hits
        processed += len(hits)
        after = hits[-1].get("sort")


def backfill_chunk_char_len(
    *,
    batch_size: int = 500,
    qdrant_batch_size: int = 200,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> dict[str, int]:
    os_client = get_client()
    qdrant_client = QdrantClient(url=QDRANT_URL)

    if dry_run:
        logger.info("Dry run enabled: skipping OpenSearch mapping update.")
    else:
        ensure_chunk_char_len_mapping()

    checksum_sort_field = _resolve_checksum_sort_field(os_client)
    stats = {
        "processed": 0,
        "os_updated": 0,
        "os_errors": 0,
        "qdrant_updated": 0,
        "qdrant_not_found": 0,
        "duplicates": 0,
        "errors": 0,
    }
    logged_sample = False

    for hits in _iter_chunks(
        os_client,
        batch_size=batch_size,
        limit=limit,
        checksum_sort_field=checksum_sort_field,
    ):
        batch_processed = 0
        batch_os_updated = 0
        batch_os_errors = 0
        batch_qdrant_updated = 0
        batch_qdrant_not_found = 0
        batch_duplicates = 0
        batch_errors = 0

        os_actions: list[dict[str, Any]] = []

        for idx, hit in enumerate(hits):
            chunk_id = hit.get("_id")
            source = hit.get("_source") or {}
            checksum = source.get("checksum")
            chunk_index_raw = source.get("chunk_index")
            if not chunk_id or checksum is None or chunk_index_raw is None:
                batch_errors += 1
                continue
            try:
                chunk_index = int(chunk_index_raw)
            except (TypeError, ValueError):
                batch_errors += 1
                continue
            text = source.get("text")
            chunk_char_len = len(text or "")

            if not logged_sample and idx < 3:
                logger.info(
                    "Sample chunk: checksum=%s chunk_index=%s chunk_char_len=%s",
                    checksum,
                    chunk_index,
                    chunk_char_len,
                )

            batch_processed += 1

            count_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="checksum", match=models.MatchValue(value=checksum)
                    ),
                    models.FieldCondition(
                        key="chunk_index", match=models.MatchValue(value=chunk_index)
                    ),
                ]
            )

            try:
                count_result = qdrant_client.count(
                    collection_name=QDRANT_COLLECTION,
                    count_filter=count_filter,
                    exact=True,
                )
                match_count = count_result.count
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to count Qdrant points checksum=%s chunk_index=%s: %s",
                    checksum,
                    chunk_index,
                    exc,
                )
                batch_errors += 1
                continue

            if match_count == 0:
                batch_qdrant_not_found += 1
            elif match_count > 1:
                batch_duplicates += 1

            if match_count > 0 and not dry_run:
                try:
                    qdrant_client.set_payload(
                        collection_name=QDRANT_COLLECTION,
                        payload={"chunk_char_len": chunk_char_len},
                        filter=count_filter,
                        wait=True,
                    )
                    batch_qdrant_updated += match_count
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to set payload for checksum=%s chunk_index=%s: %s",
                        checksum,
                        chunk_index,
                        exc,
                    )
                    batch_errors += 1

            os_actions.append(
                {
                    "_op_type": "update",
                    "_index": CHUNKS_INDEX,
                    "_id": str(chunk_id),
                    "doc": {"chunk_char_len": chunk_char_len},
                }
            )

        logged_sample = True

        if os_actions:
            if dry_run:
                logger.info(
                    "Dry run: would update %s OpenSearch docs in this batch.",
                    len(os_actions),
                )
            else:
                try:
                    success_count, errors = helpers.bulk(
                        os_client,
                        os_actions,
                        raise_on_error=False,
                    )
                    batch_os_updated += success_count
                    batch_os_errors += len(errors or [])
                except Exception as exc:  # noqa: BLE001
                    logger.error("OpenSearch bulk update failed: %s", exc)
                    batch_os_errors += len(os_actions)

        stats["processed"] += batch_processed
        stats["os_updated"] += batch_os_updated
        stats["os_errors"] += batch_os_errors
        stats["qdrant_updated"] += batch_qdrant_updated
        stats["qdrant_not_found"] += batch_qdrant_not_found
        stats["duplicates"] += batch_duplicates
        stats["errors"] += batch_errors

        logger.info(
            "Batch complete. processed=%s os_updated=%s os_errors=%s "
            "qdrant_updated=%s qdrant_not_found=%s duplicates=%s errors=%s",
            batch_processed,
            batch_os_updated,
            batch_os_errors,
            batch_qdrant_updated,
            batch_qdrant_not_found,
            batch_duplicates,
            batch_errors,
        )

    logger.info(
        "Backfill complete. processed=%s os_updated=%s os_errors=%s "
        "qdrant_updated=%s qdrant_not_found=%s duplicates=%s errors=%s",
        stats["processed"],
        stats["os_updated"],
        stats["os_errors"],
        stats["qdrant_updated"],
        stats["qdrant_not_found"],
        stats["duplicates"],
        stats["errors"],
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill chunk_char_len into Qdrant payloads."
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--qdrant-batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    backfill_chunk_char_len(
        batch_size=args.batch_size,
        qdrant_batch_size=args.qdrant_batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
