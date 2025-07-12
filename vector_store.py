import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    SearchParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    logger,
)
from sentence_transformers import SentenceTransformer


# ───────────────────────────────────────
# 🔁 Load embedding model (singleton)
# ───────────────────────────────────────
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("🔁 Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    logger.info("Embedding %d texts", len(texts))
    return model.encode(texts, normalize_embeddings=True).tolist()


# ───────────────────────────────────────
# 🔌 Qdrant client
# ───────────────────────────────────────
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(vector_size: int = 768) -> None:
    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        logger.info("Creating Qdrant collection: %s", QDRANT_COLLECTION_NAME)
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def is_file_already_indexed(checksum: str) -> bool:
    """Check if a file with the same checksum already exists in Qdrant."""
    ensure_collection()
    result = client.scroll(
        collection_name=QDRANT_COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="checksum", match=MatchValue(value=checksum))]
        ),
        limit=1,
    )
    return len(result[0]) > 0


def upsert_embeddings(
    texts: List[str], path: str, checksum: str, timestamp: Optional[str] = None
) -> None:
    """Upsert chunk vectors with metadata into Qdrant."""
    if not texts:
        return

    vectors = embed_texts(texts)
    ensure_collection(vector_size=len(vectors[0]))

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{checksum}-{i}")),
            vector=vector,
            payload={
                "path": path,
                "checksum": checksum,
                "chunk_index": i,
                "timestamp": timestamp,
                "content": text,
            },
        )
        for i, (text, vector) in enumerate(zip(texts, vectors))
    ]

    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
    logger.info("✅ Upserted %d chunks into Qdrant for %s", len(points), path)


def query_top_k(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search for top_k similar chunks to a query."""
    embedding = embed_texts([query])[0]
    ensure_collection()

    results = client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k,
        search_params=SearchParams(hnsw_ef=128),
    )

    return [
        {
            "score": result.score,
            "content": (result.payload or {}).get("content", ""),
            "metadata": {
                "path": (result.payload or {}).get("path", ""),
                "timestamp": (result.payload or {}).get("timestamp", ""),
            },
        }
        for result in results
    ]
