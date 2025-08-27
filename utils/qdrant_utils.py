from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    PointStruct,
    PointIdsList,
    VectorParams,
    Distance,
)
from config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_SIZE, logger
from typing import Optional, List, Dict, Any, Iterable, Tuple

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


def upsert_vectors(chunks: list[dict], vectors: list[list[float]]) -> bool:
    points = [
        PointStruct(
            id=chunk["id"],
            vector=vec,
            payload=chunk,
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    return True


def index_chunks(chunks: List[Dict[str, Any]]) -> bool:
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    return upsert_vectors(chunks, embeddings)


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
