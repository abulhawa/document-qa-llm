import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from config import EMBEDDING_SIZE

COLLECTION = os.getenv("QDRANT_COLLECTION", "document_chunks")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def ensure_doc_collection() -> None:
    try:
        client = QdrantClient(url=QDRANT_URL)
        client.get_collection(COLLECTION)
    except Exception:
        try:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
            )
        except Exception:
            pass
