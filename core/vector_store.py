from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
    VectorParams,
    Distance,
)
from config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    CHUNK_SCORE_THRESHOLD,
    EMBEDDING_SIZE,
    logger,
)
from core.embeddings import embed_texts
from tracing import (
    start_span,
    EMBEDDING,
    RETRIEVER,
    INPUT_VALUE,
    OUTPUT_VALUE,
    record_span_error,
    STATUS_OK,
)

# Initialize Qdrant client
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


def index_chunks(chunks: List[Dict[str, Any]]) -> bool:

    ensure_collection_exists()

    texts: List[str] = [chunk["text"] for chunk in chunks]
    try:
        embeddings = embed_texts(texts)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return False

    points = [
        PointStruct(
            id=chunk["id"],
            vector=vector,
            payload=chunk,
        )
        for vector, chunk in zip(embeddings, chunks)
    ]

    try:
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        logger.info(f"✅ Indexed {len(points)} chunks.")
        return True
    except Exception as e:
        logger.error(f"❌ Indexing to Qdrant failed: {e}")
        return False


def retrieve_top_k(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    with start_span("Semantic retriever", kind=RETRIEVER) as span:
        span.set_attribute(INPUT_VALUE, query)
        with start_span("Embed query", EMBEDDING) as espan:
            espan.set_attribute(INPUT_VALUE, query)
            espan.set_attribute("question_length", len(query))
            try:
                query_embedding = embed_texts([query])[0]
                espan.set_attribute(
                    OUTPUT_VALUE, f"{len(query_embedding)} dimensional vector"
                )
            except Exception as e:
                logger.error(f"❌ Query embedding failed: {e}")
                record_span_error(espan, e)
                return [{"status": "❌ Failed to embed query."}]

            espan.set_status(STATUS_OK)

        try:
            results = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=CHUNK_SCORE_THRESHOLD,
                with_payload=True,
            )
            retrieved_chunks = [
                {**(r.payload or {}), "score": r.score} for r in results
            ]

            for i, doc in enumerate(retrieved_chunks):
                span.set_attribute(f"retrieval.documents.{i}.document.id", doc["path"])
                span.set_attribute(
                    f"retrieval.documents.{i}.document.score", doc["score"]
                )
                span.set_attribute(
                    f"retrieval.documents.{i}.document.content", doc["text"]
                )
                span.set_attribute(
                    f"retrieval.documents.{i}.document.metadata",
                    [
                        f"Chunk index: {doc['chunk_index']}",
                        f"Date modified: {doc['modified_at']}",
                    ],
                )
            span.set_status(STATUS_OK)

            return retrieved_chunks
        except Exception as e:
            logger.error(f"Search error: {e}")
            record_span_error(span, e)
            return []
