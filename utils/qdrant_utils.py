from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    PointStruct,
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


def index_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:

    ensure_collection_exists()

    texts: List[str] = [chunk["text"] for chunk in chunks]
    try:
        embeddings = embed_texts(texts)
    except Exception:
        logger.exception("Embedding failed.")
        raise

    points = [
        PointStruct(
            id=chunk["id"],
            vector=vector,
            payload={k: v for k, v in chunk.items() if k != "has_embedding"},
        )
        for vector, chunk in zip(embeddings, chunks)
    ]

    try:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    except Exception:
        logger.exception("Qdrant upsert failed.")
        raise

    logger.info(f"✅ Indexed {len(points)} chunks.")
    return {"upserted": len(points)}


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


def delete_vectors_by_checksum(checksum: str) -> None:
    """Delete all vectors in Qdrant for a given checksum."""
    try:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="checksum", match=models.MatchValue(value=checksum)
                        )
                    ]
                )
            ),
        )
    except Exception as e:
        logger.error("❌ Qdrant delete error for checksum=%s: %s", checksum, e)


def delete_vectors_many_by_checksum(checksums: Iterable[str]) -> None:
    unique = [c for c in {c for c in checksums if c}]
    if not unique:
        return
    # Qdrant's filter supports OR via 'should'. Chunk to keep payload reasonable.
    CHUNK = 64
    for i in range(0, len(unique), CHUNK):
        part = unique[i : i + CHUNK]
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="checksum",
                    match=models.MatchAny(any=part),  # ← OR on these values
                )
            ]
        )
        try:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(filter=flt),
            )
        except Exception as e:
            logger.error(
                "❌ Qdrant batch delete error for %d checksum(s): %s", len(part), e
            )


def delete_vectors_by_path_checksum(pairs: Iterable[Tuple[str, str]]) -> None:
    """Delete vectors matching both path and checksum for each pair."""
    unique = {(p, c) for p, c in pairs if p and c}
    if not unique:
        return

    for path, checksum in unique:
        flt = models.Filter(
            must=[
                models.FieldCondition(key="path", match=models.MatchValue(value=path)),
                models.FieldCondition(
                    key="checksum", match=models.MatchValue(value=checksum)
                ),
            ]
        )
        try:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(filter=flt),
            )
        except Exception as e:
            logger.error(
                "❌ Qdrant delete error for path=%s checksum=%s: %s", path, checksum, e
            )
    logger.info(f"🗑️ Qdrant deleted vectors for {len(unique)} path/checksum pair(s).")
