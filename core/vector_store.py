from typing import List, Dict, Any
from core.opensearch_client import get_client
from utils.qdrant_utils import ensure_collection_exists
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
)
from config import (
    QDRANT_URL,
    QDRANT_COLLECTION,
    CHUNK_SCORE_THRESHOLD,
    CHUNKS_INDEX,
    logger,
)
from core.embeddings import embed_texts
from core.retrieval.types import DocHit
from tracing import (
    start_span,
    EMBEDDING,
    RETRIEVER,
    INPUT_VALUE,
    OUTPUT_VALUE,
    record_span_error,
    STATUS_OK,
)
from utils.timing import timed_block

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL)


def _fetch_chunk_texts(chunk_ids: set[str]) -> Dict[str, str]:
    """Fetch chunk text from OpenSearch given chunk IDs stored in Qdrant payloads."""

    if not chunk_ids:
        return {}

    try:
        os_client = get_client()
        with timed_block(
            "step.opensearch.query",
            extra={"index": CHUNKS_INDEX, "operation": "mget", "ids": len(chunk_ids)},
            logger=logger,
        ):
            response = os_client.mget(
                index=CHUNKS_INDEX,
                body={"ids": list(chunk_ids), "_source": ["text"]},
            )
    except Exception as e:
        logger.warning(
            "❌ OpenSearch unavailable while fetching chunk text; proceeding with Qdrant payloads."
        )
        logger.debug(e, exc_info=True)
        return {}

    texts: Dict[str, str] = {}
    for doc in response.get("docs", []):
        if not doc.get("found"):
            continue
        chunk_id = doc.get("_id")
        if chunk_id:
            texts[chunk_id] = doc.get("_source", {}).get("text", "")
    return texts


def retrieve_top_k(query: str, top_k: int = 5) -> List[DocHit]:
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
            with timed_block(
                "step.qdrant.call",
                extra={"operation": "search", "collection": QDRANT_COLLECTION, "top_k": top_k},
                logger=logger,
            ):
                results = client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=query_embedding,
                    limit=top_k * 3,
                    score_threshold=CHUNK_SCORE_THRESHOLD,
                    with_payload=True,
                )
            chunk_ids: set[str] = set()
            for r in results:
                payload = r.payload or {}
                chunk_id = payload.get("id")
                if isinstance(chunk_id, str) and chunk_id:
                    chunk_ids.add(chunk_id)
            chunk_texts = _fetch_chunk_texts(chunk_ids)
            retrieved_chunks: List[DocHit] = []
            seen_checksums = set()
            for r in results:
                payload = r.payload or {}
                checksum = payload.get("checksum")
                if checksum in seen_checksums:
                    continue
                seen_checksums.add(checksum)
                chunk_id = payload.get("id")
                if isinstance(chunk_id, str):
                    text = chunk_texts.get(chunk_id, payload.get("text", ""))
                else:
                    text = payload.get("text", "")
                retrieved_chunks.append({**payload, "score": r.score, "text": text})
                if len(retrieved_chunks) >= top_k:
                    break

            for i, doc in enumerate(retrieved_chunks):
                doc_path = doc.get("path") or ""
                doc_score = doc.get("score")
                score_value = float(doc_score) if isinstance(doc_score, (int, float)) else 0.0
                span.set_attribute(
                    f"retrieval.documents.{i}.document.id", doc_path
                )
                span.set_attribute(
                    f"retrieval.documents.{i}.document.score", score_value
                )
                span.set_attribute(
                    f"retrieval.documents.{i}.document.content", doc.get("text") or ""
                )
                span.set_attribute(
                    f"retrieval.documents.{i}.document.metadata",
                    [
                        f"Chunk index: {doc.get('chunk_index') or 'N/A'}",
                        f"Date modified: {doc.get('modified_at') or 'N/A'}",
                    ],
                )
            span.set_status(STATUS_OK)

            return retrieved_chunks
        except Exception as e:
            logger.error(f"Search error: {e}")
            record_span_error(span, e)
            return []
