# core/vector_store.py

import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
    Distance,
)
from config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    CHUNK_SCORE_THRESHOLD,
    EMBEDDING_SIZE,
    EMBEDDING_BATCH_SIZE,
    logger,
)
from core.embeddings import embed_texts
from tracing import get_tracer

tracer = get_tracer(__name__)

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL)


@tracer.start_as_current_span("ensure_collection_exists")
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


@tracer.start_as_current_span("index_chunks")
def index_chunks(texts: List[str], metadata_list: List[Dict[str, Any]]) -> bool:
    if len(texts) != len(metadata_list):
        raise ValueError("texts and metadata_list lengths diverge")

    ensure_collection_exists()

    try:
        embeddings = embed_texts(texts)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return False

    points = [
        PointStruct(
            id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS, f"{meta['checksum']}-{meta['chunk_index']}"
                )
            ),
            vector=vector,
            payload={**meta, "content": text},
        )
        for text, vector, meta in zip(texts, embeddings, metadata_list)
    ]

    try:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        logger.info(f"✅ Indexed {len(points)} chunks.")
        return True
    except Exception as e:
        logger.error(f"❌ Indexing to Qdrant failed: {e}")
        return False


@tracer.start_as_current_span("retrieve_top_k")
def retrieve_top_k(
    query_embedding: List[float], top_k: int = 5
) -> List[Dict[str, Any]]:
    try:
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=CHUNK_SCORE_THRESHOLD,
            with_payload=True,
        )
        return [{**(r.payload or {}), "score": r.score} for r in results]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


@tracer.start_as_current_span("is_file_up_to_date")
def is_file_up_to_date(checksum: str) -> bool:
    try:
        result, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="checksum", match=MatchValue(value=checksum))]
            ),
            limit=1,
            with_payload=False,
        )
        return len(result) > 0
    except Exception as e:
        logger.warning(f"Checksum check failed: {e}")
        return False


# Initialization
ensure_collection_exists()
