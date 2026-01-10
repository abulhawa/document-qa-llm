from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import uuid
from typing import Callable, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    QDRANT_FILE_VECTORS_COLLECTION,
    EMBEDDING_SIZE,
    logger,
)
from utils.timing import timed_block


SCROLL_BATCH_SIZE = 256
UPDATE_EVERY = 10


client = QdrantClient(url=QDRANT_URL)


@dataclass
class BuildStats:
    processed: int = 0
    created: int = 0
    skipped: int = 0
    errors: int = 0
    total: int = 0
    error_list: List[str] | None = None

    def as_dict(self) -> dict:
        return {
            "processed": self.processed,
            "created": self.created,
            "skipped": self.skipped,
            "errors": self.errors,
            "total": self.total,
            "error_list": self.error_list or [],
        }


def _collection_exists(collection_name: str) -> bool:
    with timed_block(
        "step.qdrant.call",
        extra={"operation": "get_collections", "collection": collection_name},
        logger=logger,
    ):
        collections = client.get_collections().collections
    return collection_name in [c.name for c in collections]


def ensure_file_vectors_collection(
    dim: int = EMBEDDING_SIZE,
    distance: str = "Cosine",
    *,
    recreate: bool = False,
) -> None:
    if _collection_exists(QDRANT_FILE_VECTORS_COLLECTION):
        if recreate:
            logger.info("Recreating Qdrant collection '%s'", QDRANT_FILE_VECTORS_COLLECTION)
            with timed_block(
                "step.qdrant.call",
                extra={"operation": "delete_collection", "collection": QDRANT_FILE_VECTORS_COLLECTION},
                logger=logger,
            ):
                client.delete_collection(collection_name=QDRANT_FILE_VECTORS_COLLECTION)
        else:
            logger.info("Qdrant collection '%s' exists.", QDRANT_FILE_VECTORS_COLLECTION)
            return

    logger.info("Creating Qdrant collection '%s'", QDRANT_FILE_VECTORS_COLLECTION)
    distance_value = models.Distance.COSINE
    if distance.lower() == "dot":
        distance_value = models.Distance.DOT
    elif distance.lower() == "euclid":
        distance_value = models.Distance.EUCLID
    with timed_block(
        "step.qdrant.call",
        extra={"operation": "create_collection", "collection": QDRANT_FILE_VECTORS_COLLECTION},
        logger=logger,
    ):
        client.create_collection(
            collection_name=QDRANT_FILE_VECTORS_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=distance_value),
        )


def _scroll_collection(
    *,
    collection_name: str,
    scroll_filter: models.Filter | None = None,
    with_vectors: bool = False,
    with_payload: bool = True,
    batch_size: int = SCROLL_BATCH_SIZE,
) -> Iterable[models.Record]:
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            with_vectors=with_vectors,
            with_payload=with_payload,
            offset=offset,
        )
        for point in points:
            yield point
        if offset is None:
            break


def get_unique_checksums_in_chunks() -> set[str]:
    checksums: set[str] = set()
    with timed_block(
        "step.qdrant.call",
        extra={"operation": "scroll_unique_checksums", "collection": QDRANT_COLLECTION},
        logger=logger,
    ):
        for point in _scroll_collection(collection_name=QDRANT_COLLECTION, with_vectors=False, with_payload=True):
            payload = _extract_payload(point)
            checksum = payload.get("checksum")
            if checksum:
                checksums.add(str(checksum))
    return checksums


def get_file_vectors_count() -> int:
    if not _collection_exists(QDRANT_FILE_VECTORS_COLLECTION):
        return 0
    with timed_block(
        "step.qdrant.call",
        extra={"operation": "count", "collection": QDRANT_FILE_VECTORS_COLLECTION},
        logger=logger,
    ):
        result = client.count(
            collection_name=QDRANT_FILE_VECTORS_COLLECTION,
            exact=True,
        )
    return int(result.count)


def get_existing_file_vector_checksums() -> set[str]:
    if not _collection_exists(QDRANT_FILE_VECTORS_COLLECTION):
        return set()
    checksums: set[str] = set()
    with timed_block(
        "step.qdrant.call",
        extra={
            "operation": "scroll_existing_vectors",
            "collection": QDRANT_FILE_VECTORS_COLLECTION,
        },
        logger=logger,
    ):
        for point in _scroll_collection(
            collection_name=QDRANT_FILE_VECTORS_COLLECTION,
            with_vectors=False,
            with_payload=True,
        ):
            payload = _extract_payload(point)
            checksum = payload.get("checksum")
            if not checksum:
                checksum = _extract_id(point)
            if checksum:
                checksums.add(str(checksum))
    return checksums


def build_missing_file_vectors(
    k: int = 8,
    batch: int = 64,
    limit: int | None = None,
    force: bool = False,
    *,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    ensure_file_vectors_collection()
    all_checksums = sorted(get_unique_checksums_in_chunks())
    existing_checksums = set() if force else get_existing_file_vector_checksums()
    target_checksums = [c for c in all_checksums if c not in existing_checksums]

    if limit is not None:
        target_checksums = target_checksums[:limit]

    stats = BuildStats(total=len(target_checksums), error_list=[])
    points: List[models.PointStruct] = []

    for checksum in target_checksums:
        stats.processed += 1
        try:
            chunk_vectors, payloads = _fetch_chunk_vectors_by_checksum(checksum)
            if not chunk_vectors:
                stats.skipped += 1
                _append_error(stats, f"{checksum}: no vectors found")
                _maybe_report(progress_callback, stats, checksum, "skipped")
                continue

            file_vector = _topk_mean_pool(
                chunk_vectors,
                payloads,
                k=k,
            )
            payload = {
                "checksum": checksum,
                "n_chunks": len(chunk_vectors),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            points.append(
                models.PointStruct(
                    id=_file_vector_id(checksum),
                    vector=file_vector,
                    payload=payload,
                )
            )
            stats.created += 1
        except Exception as exc:  # noqa: BLE001
            stats.errors += 1
            _append_error(stats, f"{checksum}: {exc}")
        _maybe_report(progress_callback, stats, checksum, "processed")

        if len(points) >= batch:
            _upsert_points(points, stats)
            points = []

    if points:
        _upsert_points(points, stats)

    return stats.as_dict()


def sample_file_vectors(limit: int = 5) -> list[dict]:
    if not _collection_exists(QDRANT_FILE_VECTORS_COLLECTION):
        return []
    sampled: list[dict] = []
    for point in _scroll_collection(
        collection_name=QDRANT_FILE_VECTORS_COLLECTION,
        with_vectors=True,
        with_payload=True,
        batch_size=limit,
    ):
        vector = _extract_vector(point)
        if not vector:
            continue
        checksum = _extract_payload(point).get("checksum") or _extract_id(point)
        if not checksum:
            continue
        sampled.append(
            {
                "checksum": str(checksum),
                "dim": len(vector),
                "norm": _l2_norm(vector),
            }
        )
        if len(sampled) >= limit:
            break
    return sampled


def _fetch_chunk_vectors_by_checksum(
    checksum: str,
) -> tuple[list[list[float]], list[dict]]:
    scroll_filter = models.Filter(
        must=[models.FieldCondition(key="checksum", match=models.MatchValue(value=checksum))]
    )
    vectors: list[list[float]] = []
    payloads: list[dict] = []
    for point in _scroll_collection(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=scroll_filter,
        with_vectors=True,
        with_payload=True,
    ):
        vector = _extract_vector(point)
        if vector:
            vectors.append(vector)
            payloads.append(_extract_payload(point))
    return vectors, payloads


def _extract_payload(point: models.Record) -> dict:
    if isinstance(point, dict):
        return point.get("payload", {}) or {}
    payload = getattr(point, "payload", None)
    return payload or {}


def _extract_vector(point: models.Record) -> list[float] | None:
    vector = None
    if isinstance(point, dict):
        vector = point.get("vector")
    else:
        vector = getattr(point, "vector", None)
    if vector is None:
        return None
    if isinstance(vector, dict):
        if vector:
            value = next(iter(vector.values()))
            if value is None:
                return None
            return list(value)
        return None
    return list(vector)


def _extract_id(point: models.Record) -> str | None:
    if isinstance(point, dict):
        return str(point.get("id")) if point.get("id") is not None else None
    point_id = getattr(point, "id", None)
    return str(point_id) if point_id is not None else None


def _file_vector_id(checksum: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"file-vector:{checksum}"))


def _topk_mean_pool(
    vectors: list[list[float]],
    payloads: list[dict],
    *,
    k: int,
) -> list[float]:
    normalized = [_l2_normalize(vec) for vec in vectors]
    scored: list[tuple[float, list[float]]] = []
    for vec, payload in zip(normalized, payloads):
        raw_score = payload.get("chunk_char_len")
        if isinstance(raw_score, (int, float, str)):
            try:
                score = float(raw_score)
            except ValueError:
                score = 1.0
        else:
            score = 1.0
        if score <= 0:
            score = 1.0
        scored.append((score, vec))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_vectors = [vec for _, vec in scored[: max(1, k)]]

    mean_vec = [0.0 for _ in range(len(top_vectors[0]))]
    for vec in top_vectors:
        if len(vec) != len(mean_vec):
            raise ValueError("Vector size mismatch while pooling")
        for idx, val in enumerate(vec):
            mean_vec[idx] += val
    count = float(len(top_vectors))
    mean_vec = [val / count for val in mean_vec]
    return _l2_normalize(mean_vec)


def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(val * val for val in vec))


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = _l2_norm(vec)
    if norm == 0.0:
        return [0.0 for _ in vec]
    return [val / norm for val in vec]


def _upsert_points(points: list[models.PointStruct], stats: BuildStats) -> None:
    try:
        with timed_block(
            "step.qdrant.call",
            extra={
                "operation": "upsert",
                "collection": QDRANT_FILE_VECTORS_COLLECTION,
                "points": len(points),
            },
            logger=logger,
        ):
            client.upsert(
                collection_name=QDRANT_FILE_VECTORS_COLLECTION,
                points=points,
                wait=True,
            )
    except Exception as exc:  # noqa: BLE001
        stats.errors += len(points)
        _append_error(stats, f"batch upsert failed: {exc}")


def _append_error(stats: BuildStats, message: str) -> None:
    if stats.error_list is None:
        stats.error_list = []
    if len(stats.error_list) < 10:
        stats.error_list.append(message)


def _maybe_report(
    progress_callback: Callable[[dict], None] | None,
    stats: BuildStats,
    checksum: str,
    status: str,
) -> None:
    if not progress_callback:
        return
    if stats.processed % UPDATE_EVERY != 0 and stats.processed != stats.total:
        return
    progress_callback(
        {
            "checksum": checksum,
            "status": status,
            "processed": stats.processed,
            "created": stats.created,
            "skipped": stats.skipped,
            "errors": stats.errors,
            "total": stats.total,
        }
    )
