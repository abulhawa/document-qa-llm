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
from typing import Optional, List, Dict, Any, Iterable, Sequence, Callable

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
