"""
utils/qdrant_utils.py
=====================
Low-level Qdrant operations: collection management, vector upsert, search,
and deletion. All higher-level retrieval logic lives in core/vector_store.py.

Payload design philosophy
--------------------------
Qdrant stores vectors + a small payload per point. We deliberately keep the
payload minimal (_QDRANT_PAYLOAD_KEYS = id, checksum, path) because:

  1. OpenSearch is the single source of truth for all chunk metadata.
     After ANN search, core/vector_store.py fetches full metadata from
     OpenSearch via _fetch_chunk_texts(). Nothing else needs to be in Qdrant.

  2. Keeping payloads lean means metadata changes (doc_type corrections,
     financial enrichment re-runs, new fields) only require an OpenSearch
     update. No Qdrant backfill needed.

  3. Smaller payloads = less memory pressure on Qdrant at scale.

Migration note
--------------
Points ingested before this payload-slim change (any point whose payload
contains keys beyond id/checksum/path) can be cleaned up without re-embedding
by running:

    python scripts/slim_qdrant_payloads.py --dry-run
    python scripts/slim_qdrant_payloads.py

This script scrolls all points, detects bloated payloads, and overwrites them
with the minimal set using client.set_payload() + client.clear_payload().
Vectors are never touched.
"""

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
from utils.timing import timed_block

client = QdrantClient(url=QDRANT_URL)


def ensure_collection_exists() -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Called once at startup (app/main.py) and before ingestion runs.
    Uses HNSW with cosine distance, sized to EMBEDDING_SIZE from config.
    """
    with timed_block(
        "step.qdrant.call",
        extra={"operation": "get_collections", "collection": QDRANT_COLLECTION},
        logger=logger,
    ):
        collections = client.get_collections().collections
    if QDRANT_COLLECTION in [c.name for c in collections]:
        logger.info(f"Collection '{QDRANT_COLLECTION}' exists.")
        return

    logger.info(f"Creating collection '{QDRANT_COLLECTION}'...")
    with timed_block(
        "step.qdrant.call",
        extra={"operation": "create_collection", "collection": QDRANT_COLLECTION},
        logger=logger,
    ):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )
    logger.info(f"Created collection '{QDRANT_COLLECTION}'.")


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

# Minimal payload stored in Qdrant alongside each vector.
# Qdrant is used exclusively for ANN search — all metadata lives in OpenSearch.
# We only store what is needed to:
#   - look up the chunk in OpenSearch after retrieval (id)
#   - deduplicate results in retrieve_top_k() (checksum)
#   - support deletion by path (path)
# Everything else (doc_type, financial metadata, chunk_index, etc.) is fetched
# from OpenSearch via _fetch_chunk_texts() and never needs to live here.
# Benefit: metadata backfills (e.g. re-running financial enrichment) only
# require an OpenSearch update — no Qdrant pass needed.
_QDRANT_PAYLOAD_KEYS = {"id", "checksum", "path"}


def upsert_vectors(chunks: list[dict], vectors: list[list[float]]) -> bool:
    points = [
        PointStruct(
            id=chunk["id"],
            vector=vec,
            payload={k: chunk[k] for k in _QDRANT_PAYLOAD_KEYS if k in chunk},
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    with timed_block(
        "step.qdrant.call",
        extra={"operation": "upsert", "collection": QDRANT_COLLECTION, "points": len(points)},
        logger=logger,
    ):
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
    Embed and upsert a list of chunks in batches, then optionally index in OpenSearch.

    Vectors-first ordering per batch is intentional:
      1) embed the batch
      2) upsert to Qdrant (wait=True — blocks until persisted)
      3) call os_index_batch for the same batch

    This ensures Qdrant never falls behind OpenSearch. If the process dies
    mid-batch, the worst case is a Qdrant point with no corresponding OpenSearch
    doc (retrieval returns it but fetch fails gracefully), never the reverse.

    Called by ingestion/storage.py:embed_and_store().
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


def count_qdrant_chunks_by_checksum(checksum: str) -> Optional[int]:
    """
    Return the number of chunks in Qdrant matching the given checksum.
    """
    try:
        with timed_block(
            "step.qdrant.call",
            extra={"operation": "count", "collection": QDRANT_COLLECTION},
            logger=logger,
        ):
            result = client.count(
                collection_name=QDRANT_COLLECTION,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="checksum", match=models.MatchValue(value=checksum)
                        ),
                    ]
                ),
                exact=True,
            )
        return result.count
    except Exception as e:
        logger.error("❌ Qdrant count error for checksum=%s: %s", checksum, e)
        return None


def delete_vectors_by_ids(ids: list[str]) -> int:
    """
    Delete Qdrant points by their chunk ID list.
    Called by ingestion/storage.py:replace_existing_artifacts() before re-ingestion.
    Returns the number of IDs requested for deletion (not confirmed deleted).
    """
    if not ids:
        return 0
    with timed_block(
        "step.qdrant.call",
        extra={"operation": "delete_by_ids", "collection": QDRANT_COLLECTION, "points": len(ids)},
        logger=logger,
    ):
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=PointIdsList(points=[i for i in ids]),
            wait=True,
        )
    return len(ids)


def delete_vectors_by_checksum(checksum: str) -> int:
    """
    Delete all Qdrant points matching a document checksum.
    Used as an alternative to delete_vectors_by_ids when chunk IDs are not
    known but the file checksum is (e.g. cross-path duplicate handling).
    Returns the number of points deleted, or 0 on error.
    """
    if not checksum:
        return 0
    try:
        with timed_block(
            "step.qdrant.call",
            extra={"operation": "delete_by_checksum", "collection": QDRANT_COLLECTION},
            logger=logger,
        ):
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
