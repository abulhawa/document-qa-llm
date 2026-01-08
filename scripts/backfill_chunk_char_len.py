from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, cast

from core.opensearch_client import get_client
from qdrant_client import QdrantClient
from qdrant_client.http.models import ExtendedPointId, PointIdsList

from config import CHUNKS_INDEX, QDRANT_COLLECTION, QDRANT_URL, logger


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
            "_source": ["checksum", "chunk_index", "chunk_char_len"],
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


def _build_qdrant_lookup(
    qdrant_client: QdrantClient,
    *,
    scroll_batch_size: int,
) -> tuple[dict[tuple[str, int], list[str | int]], int, int]:
    lookup: dict[tuple[str, int], list[str | int]] = defaultdict(list)
    duplicates = 0
    errors = 0
    offset = None

    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=scroll_batch_size,
            with_payload=["checksum", "chunk_index"],
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            payload = point.payload or {}
            checksum = payload.get("checksum")
            chunk_index_raw = payload.get("chunk_index")
            point_id = point.id
            if checksum is None or chunk_index_raw is None or point_id is None:
                errors += 1
                continue
            try:
                chunk_index = int(chunk_index_raw)
            except (TypeError, ValueError):
                errors += 1
                continue
            key = (checksum, chunk_index)
            if lookup[key]:
                duplicates += 1
            lookup[key].append(point_id)

        offset = next_offset
        if offset is None:
            break

    return lookup, duplicates, errors


def _batched_ids(ids: list[str | int], size: int) -> Iterable[list[str | int]]:
    for start in range(0, len(ids), size):
        yield ids[start : start + size]


def backfill_chunk_char_len(
    *,
    batch_size: int = 500,
    qdrant_batch_size: int = 1000,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> dict[str, int]:
    os_client = get_client()
    qdrant_client = QdrantClient(url=QDRANT_URL)

    checksum_sort_field = _resolve_checksum_sort_field(os_client)
    qdrant_lookup, qdrant_duplicates, lookup_errors = _build_qdrant_lookup(
        qdrant_client,
        scroll_batch_size=batch_size,
    )
    stats = {
        "processed_os": 0,
        "qdrant_updated_points": 0,
        "not_found": 0,
        "already_migrated": 0,
        "duplicates": qdrant_duplicates,
        "errors": lookup_errors,
    }

    for hits in _iter_chunks(
        os_client,
        batch_size=batch_size,
        limit=limit,
        checksum_sort_field=checksum_sort_field,
    ):
        batch_processed = 0
        batch_qdrant_updated = 0
        batch_qdrant_not_found = 0
        batch_errors = 0
        grouped_updates: dict[int, list[str | int]] = defaultdict(list)

        for hit in hits:
            source = hit.get("_source") or {}
            checksum = source.get("checksum")
            chunk_index_raw = source.get("chunk_index")
            chunk_char_len = source.get("chunk_char_len")
            if checksum is None or chunk_index_raw is None:
                batch_errors += 1
                continue
            try:
                chunk_index = int(chunk_index_raw)
            except (TypeError, ValueError):
                batch_errors += 1
                continue
            if chunk_char_len is None:
                batch_errors += 1
                continue
            try:
                chunk_char_len = int(chunk_char_len)
            except (TypeError, ValueError):
                batch_errors += 1
                continue

            batch_processed += 1
            key = (checksum, chunk_index)
            point_ids = qdrant_lookup.get(key)

            if not point_ids:
                batch_qdrant_not_found += 1
                continue

            grouped_updates[chunk_char_len].extend(point_ids)

        for chunk_char_len, point_ids in grouped_updates.items():
            for ids_batch in _batched_ids(point_ids, qdrant_batch_size):
                if dry_run:
                    logger.info(
                        "Dry run: would update %s points for chunk_char_len=%s.",
                        len(ids_batch),
                        chunk_char_len,
                    )
                    continue
                try:
                    ids_batch_typed = cast(list[ExtendedPointId], ids_batch)
                    points_selector = PointIdsList(points=ids_batch_typed)
                    qdrant_client.set_payload(
                        collection_name=QDRANT_COLLECTION,
                        payload={"chunk_char_len": chunk_char_len},
                        points=points_selector,
                        wait=False,
                    )
                    batch_qdrant_updated += len(ids_batch)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to set payload for chunk_char_len=%s: %s",
                        chunk_char_len,
                        exc,
                    )
                    batch_errors += len(ids_batch)

        stats["processed_os"] += batch_processed
        stats["qdrant_updated_points"] += batch_qdrant_updated
        stats["not_found"] += batch_qdrant_not_found
        stats["errors"] += batch_errors

        logger.info(
            "Batch complete. processed_os=%s qdrant_updated_points=%s not_found=%s "
            "already_migrated=%s duplicates=%s errors=%s",
            batch_processed,
            batch_qdrant_updated,
            batch_qdrant_not_found,
            stats["already_migrated"],
            stats["duplicates"],
            batch_errors,
        )

    logger.info(
        "Backfill complete. processed_os=%s qdrant_updated_points=%s not_found=%s "
        "already_migrated=%s duplicates=%s errors=%s",
        stats["processed_os"],
        stats["qdrant_updated_points"],
        stats["not_found"],
        stats["already_migrated"],
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
