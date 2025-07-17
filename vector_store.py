import requests
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
    CHUNK_SCORE_THRESHOLD,
    logger,
    EMBEDDING_API_URL,
)


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    logger.info(f"Embedding {len(texts)} texts via API...")
    try:
        response = requests.post(
            EMBEDDING_API_URL,
            json={"texts": texts, "batch_size": batch_size},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
    except requests.RequestException as e:
        logger.error("Embedding API request failed: %s", str(e))
        raise RuntimeError(f"Embedding API error: {e}")


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
    texts: List[str], metadata_list: List[Dict[str, Any]]
) -> None:
    """Upsert chunk vectors with metadata into Qdrant."""
    if not texts:
        return

    vectors = embed_texts(texts)
    ensure_collection(vector_size=len(vectors[0]))

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{meta['checksum']}-{meta['chunk_index']}")),
            vector=vector,
            payload={**meta, "content": text},
        )
        for i, (text, vector, meta) in enumerate(zip(texts, vectors, metadata_list))
    ]

    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
    logger.info("✅ Upserted %d chunks into Qdrant for %s", len(points), metadata_list[0]["path"])


def query_top_k(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Search for top_k similar chunks to a query."""
    embedding = embed_texts([query])[0]
    ensure_collection()

    print(f"🔍 Searching for top {top_k} chunks similar to: {query}")

    results = client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k,
        score_threshold=CHUNK_SCORE_THRESHOLD,
        search_params=SearchParams(hnsw_ef=128),
    )

    return [
        {
            "score": result.score,
            "content": (result.payload or {}).get("content", ""),
            "metadata": {
                "path": (result.payload or {}).get("path", ""),
                "timestamp": (result.payload or {}).get("timestamp", ""),
                "page": (result.payload or {}).get("page"),
                "location_percent": (result.payload or {}).get("location_percent"),
            },
        }
        for result in results
    ]
