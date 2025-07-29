import json
from typing import List, Dict, Optional
from config import logger
from datetime import datetime
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from embeddings import embed_texts
from tracing import start_span, INPUT_VALUE, OUTPUT_VALUE, record_span_error, TOOL, RETRIEVER

client = weaviate.connect_to_local()

VECTOR_COLLECTION = "DocumentChunk"
BM25_COLLECTION = "DocumentText"

# ─────────────────────────────────────────────────────────────
# Schema Creation
# ─────────────────────────────────────────────────────────────


def init_vector_collection():
    with start_span("init_vector_collection", kind=TOOL) as span:
        try:
            if VECTOR_COLLECTION in client.collections.list_all().keys():
                logger.info(f"✅ Collection '{VECTOR_COLLECTION}' already exists.")
                return
            client.collections.create(
                name=VECTOR_COLLECTION,
                vectorizer_config=None,
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="path", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="location_percent", data_type=DataType.NUMBER),
                    Property(name="created_at", data_type=DataType.DATE),
                    Property(name="modified_at", data_type=DataType.DATE),
                ],
            )
            logger.info(f"✅ Created collection '{VECTOR_COLLECTION}'.")
        except Exception as e:
            logger.error(f"❌ Failed to create vector schema: {e}")
            record_span_error(span, e)
            raise


def init_bm25_collection():
    with start_span("init_bm25_collection", kind=TOOL) as span:
        try:
            if BM25_COLLECTION in client.collections.list_all().keys():
                logger.info(f"✅ Collection '{BM25_COLLECTION}' already exists.")
                return
            client.collections.create(
                name=BM25_COLLECTION,
                vector_config=None,
                properties=[
                    Property(name="full_text", data_type=DataType.TEXT),
                    Property(name="path", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE),
                    Property(name="modified_at", data_type=DataType.DATE),
                ],
            )
            logger.info(f"✅ Created collection '{BM25_COLLECTION}'.")
        except Exception as e:
            logger.error(f"❌ Failed to create BM25 schema: {e}")
            record_span_error(span, e)
            raise


def init_weaviate_schema():
    init_vector_collection()
    init_bm25_collection()


# ─────────────────────────────────────────────────────────────
# Indexing
# ─────────────────────────────────────────────────────────────


def index_chunks(chunks: List[Dict]) -> None:
    with start_span("index_chunks", kind=TOOL) as span:
        span.set_attribute("chunk_count", len(chunks))

        if not chunks:
            logger.warning("⚠️ No chunks to index.")
            return

        try:
            texts = [chunk["content"] for chunk in chunks]
            span.set_attribute(INPUT_VALUE, texts[:3])
            vectors = embed_texts(texts)

            if len(vectors) != len(chunks):
                logger.error("❌ Mismatch between embeddings and chunks.")
                span.set_attribute("embedding_error", True)
                return

            collection = client.collections.get(VECTOR_COLLECTION)
            with collection.batch.dynamic() as batch:
                for chunk, vector in zip(chunks, vectors):
                    batch.add_object(
                        properties={
                            "content": chunk["content"],
                            "path": chunk["path"],
                            "page": chunk.get("page"),
                            "chunk_index": chunk["chunk_index"],
                            "location_percent": chunk.get("location_percent"),
                            "created_at": chunk["created_at"],
                            "modified_at": chunk["modified_at"],
                        },
                        vector=vector,
                    )
                span.set_attribute("failed_objects", batch.number_errors)
            logger.info(f"✅ Indexed {len(chunks)} chunks into '{VECTOR_COLLECTION}'.")
            span.set_attribute(OUTPUT_VALUE, json.dumps({"indexed_count": len(chunks)}))

        except Exception as e:
            logger.error(f"❌ Exception during indexing: {e}")
            record_span_error(span, e)
            raise


def index_fulltext_documents(docs: List[Dict]) -> None:
    with start_span("index_fulltext_documents", kind=TOOL) as span:
        span.set_attribute("document_count", len(docs))

        try:
            collection = client.collections.get(BM25_COLLECTION)
            with collection.batch.dynamic() as batch:
                for doc in docs:
                    batch.add_object(
                        properties={
                            "full_text": doc["full_text"],
                            "path": doc["path"],
                            "created_at": doc["created_at"],
                            "modified_at": doc["modified_at"],
                        }
                    )
                span.set_attribute("failed_objects", batch.number_errors)
            logger.info(
                f"✅ Indexed {len(docs)} full-text documents into '{BM25_COLLECTION}'."
            )
            span.set_attribute(OUTPUT_VALUE, json.dumps({"indexed_count": len(docs)}))

        except Exception as e:
            logger.error(f"❌ Exception during full-text indexing: {e}")
            record_span_error(span, e)
            raise


# ─────────────────────────────────────────────────────────────
# Retrieval and Fusion (Manual)
# ─────────────────────────────────────────────────────────────


def hybrid_retrieve(query: str, top_k: int = 10, alpha: float = 0.5) -> List[Dict]:
    from weaviate.classes.query import HybridFusion

    with start_span("hybrid_retrieve", kind=RETRIEVER) as span:
        span.set_attribute(INPUT_VALUE, query)

        try:
            collection = client.collections.get(VECTOR_COLLECTION)
            results = (
                collection.query.hybrid(
                    query=query,
                    alpha=alpha,
                    limit=top_k,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                )
                .with_additional(["score", "certainty"])
                .do()
            )

            output = []
            for o in results.objects:
                prop = o.properties
                prop["_score"] = o.score
                prop["_certainty"] = o.certainty
                output.append(prop)

            span.set_attribute("retrieved_count", len(output))
            span.set_attribute(OUTPUT_VALUE, output[:3])
            return output

        except Exception as e:
            logger.error(f"❌ Hybrid retrieval failed: {e}")
            record_span_error(span, e)
            return []
