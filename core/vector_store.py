from typing import List, Dict, Any
from utils.qdrant_utils import ensure_collection_exists
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
)
from config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    CHUNK_SCORE_THRESHOLD,
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
                limit=top_k * 3,
                score_threshold=CHUNK_SCORE_THRESHOLD,
                with_payload=True,
            )
            retrieved_chunks = []
            seen_checksums = set()
            for r in results:
                payload = r.payload or {}
                checksum = payload.get("checksum")
                if checksum in seen_checksums:
                    continue
                seen_checksums.add(checksum)
                retrieved_chunks.append({**payload, "score": r.score})
                if len(retrieved_chunks) >= top_k:
                    break

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
