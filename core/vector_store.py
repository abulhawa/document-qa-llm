import requests
from tracing import get_tracer
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from core.embeddings import embed_texts
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
    QDRANT_URL,
    QDRANT_COLLECTION,
    CHUNK_SCORE_THRESHOLD,
    logger,
)

# initialize tracer
tracer = get_tracer(__name__)

# ───────────────────────────────────────
# 🔌 Qdrant client
# ───────────────────────────────────────
client = QdrantClient(url=QDRANT_URL)

@tracer.chain
def ensure_collection(vector_size: int = 768) -> None:
    if not client.collection_exists(QDRANT_COLLECTION):
        logger.info("Creating Qdrant collection: %s", QDRANT_COLLECTION)
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

@tracer.chain
def is_file_already_indexed(checksum: str) -> bool:
    """Check if a file with the same checksum already exists in Qdrant."""
    ensure_collection()
    result = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="checksum", match=MatchValue(value=checksum))]
        ),
        limit=1,
    )
    return len(result[0]) > 0

@tracer.chain
def index_chunks(
    texts: List[str], metadata_list: List[Dict[str, Any]]
) -> bool:
    """Upsert chunk vectors with metadata into Qdrant."""
    if not texts:
        return False

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
    
    try:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        logger.info(f"✅ Indexed {len(points)} chunks into Qdrant for {metadata_list[0]['path']}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to index chunks: {e}")
        return False

@tracer.chain
def query_top_k(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Search for top_k similar chunks to a query."""
    embedding = embed_texts([query])[0]
    ensure_collection()

    print(f"🔍 Searching for top {top_k} chunks similar to: {query}")

    results = client.search(
        collection_name=QDRANT_COLLECTION,
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
