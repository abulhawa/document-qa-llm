from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    PointStruct,
    PointIdsList,
    VectorParams,
    Distance,
)
from config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    EMBEDDING_SIZE,
    EMBEDDING_REQ_MAX_CHUNKS,
    EMBEDDING_BATCH_SIZE,
    logger,
)
import math
from typing import Optional, List, Dict, Any, Iterable, Sequence, Callable, cast, Tuple

from core.embeddings import embed_texts

client = QdrantClient(url=QDRANT_URL)


def ensure_collection_exists() -> None:
    collections = client.get_collections().collections
    if QDRANT_COLLECTION in [c.name for c in collections]:
        logger.info(f"Collection '{QDRANT_COLLECTION}' exists.")
        return

    logger.info(f"Creating collection '{QDRANT_COLLECTION}'...")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )
    logger.info(f"Created collection '{QDRANT_COLLECTION}'.")

def _payload_without_text(chunk: Dict[str, Any]) -> Dict[str, Any]:
    # copy to avoid mutating the original
    p = dict(chunk)
    p.pop("text", None)   # <- drop the heavy field
    return p


def _sanitize_vectors(
    vectors: List[List[float]],
    *,
    expected_size: int,
) -> Tuple[List[List[float]], int]:
    """
    Ensure vectors are JSON-safe (finite floats) and correct length.

    Returns (possibly copied vectors, num_replacements).
    """
    replacements = 0
    any_changed = False
    sanitized: List[List[float]] = []

    for vec in vectors:
        if len(vec) != expected_size:
            raise ValueError(
                f"Embedding size mismatch: expected {expected_size}, got {len(vec)}"
            )
        new_vec = []
        changed = False
        for val in vec:
            fval = float(val)
            if not math.isfinite(fval):
                fval = 0.0
                replacements += 1
                changed = True
            new_vec.append(fval)
        if changed:
            any_changed = True
        sanitized.append(new_vec)

    if not any_changed:
        return vectors, 0
    return sanitized, replacements

def upsert_vectors(chunks: list[dict], vectors: list[list[float]]) -> bool:
    points = [
        PointStruct(
            id=chunk["id"],
            vector=vec,
            payload={k: val for k, val in chunk.items() if k != "text"},
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    return True


def _batches_by_budget(
    chunks: Sequence[Dict[str, Any]],
    max_chunks: int,
) -> Iterable[Sequence[Dict[str, Any]]]:
    """Yield fixed-size groups of chunks (no char budget, no splitting logic)."""
    for i in range(0, len(chunks), max_chunks):
        yield chunks[i : i + max_chunks]

def index_chunks_in_batches(
    chunks: List[Dict[str, Any]],
    os_index_batch: Callable[[Sequence[Dict[str, Any]]], None] | None = None,
) -> bool:
    """
    Vectors-first per batch:
      1) embed a batch
      2) upsert to Qdrant (wait=True)
      3) optionally index the same batch in OpenSearch
    """
    for group in _batches_by_budget(chunks, EMBEDDING_REQ_MAX_CHUNKS):
        texts = [c["text"] for c in group]
        vectors = embed_texts(texts, batch_size=EMBEDDING_BATCH_SIZE)
        vectors, replaced = _sanitize_vectors(vectors, expected_size=EMBEDDING_SIZE)
        if replaced:
            logger.warning(
                "Replaced %s non-finite embedding values with 0.0 before Qdrant upsert.",
                replaced,
            )
        upsert_vectors(list(group), vectors)        # blocks until persisted
        if os_index_batch:
            os_index_batch(group)                   # OS never outruns vectors
    return True


def count_qdrant_chunks_by_path(path: str) -> Optional[int]:
    """
    Return the number of chunks in Qdrant matching the given path.
    """
    try:
        result = client.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="path", match=models.MatchValue(value=path)
                    ),
                ]
            ),
            exact=True,
        )
        return result.count
    except Exception as e:
        logger.error("❌ Qdrant count error for path=%s: %s", path, e)
        return None


def delete_vectors_by_ids(ids: list[str]) -> int:
    """Delete Qdrant points by chunk IDs. Returns number requested."""
    if not ids:
        return 0
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=PointIdsList(points=[i for i in ids]),
        wait=True,
    )
    return len(ids)


def delete_vectors_by_checksum(checksum: str) -> int:
    """Delete Qdrant points by checksum filter."""
    if not checksum:
        return 0
    try:
        result = client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="checksum", match=models.MatchValue(value=checksum))]
            ),
            wait=True,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Qdrant delete failed for checksum=%s: %s", checksum, e)
        return 0
    result_any = cast(Any, result)
    if isinstance(result_any, dict):
        return int(result_any.get("result", {}).get("points_count", 0))
    result_payload = getattr(result_any, "result", None)
    if isinstance(result_payload, dict):
        return int(result_payload.get("points_count", 0))
    model_dump = getattr(result_any, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return int(dumped.get("result", {}).get("points_count", 0))
    return 0
