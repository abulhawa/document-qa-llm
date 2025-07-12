import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from db import get_all_chunks_with_embeddings, get_chunk_by_ids
from config import logger


SKIP_EMBEDDINGS = os.getenv("DOCQA_TEST_MODE") == "1"

# Initialize embedding model
if SKIP_EMBEDDINGS:
    # If in test mode, skip real embeddings to speed up tests
    logger.warning("🔧 TEST MODE: Skipping real embeddings")
    model = None
else:
    logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)


# FAISS index path
INDEX_PATH = config.FAISS_INDEX_PATH


def cosine_to_faiss(vectors):
    """FAISS requires normalized vectors for cosine similarity."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norm


def create_faiss_index(dim):
    logger.info("Creating new FAISS index (cosine similarity)")
    return faiss.IndexIDMap(faiss.IndexFlatIP(dim))


def save_index(index):
    os.makedirs(os.path.dirname(config.FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    logger.info("FAISS index saved to disk at: %s", INDEX_PATH)


def load_index():
    if os.path.exists(INDEX_PATH):
        logger.info("Loading FAISS index from disk: %s", INDEX_PATH)
        return faiss.read_index(INDEX_PATH)
    else:
        logger.warning("No existing FAISS index found.")
        return None


def rebuild_faiss_index():
    logger.info("Rebuilding FAISS index from all stored embeddings in DB...")
    chunks = get_all_chunks_with_embeddings()
    if not chunks:
        logger.warning("No chunks with embeddings found in DB.")
        return None

    embeddings = np.array([c["embedding"] for c in chunks]).astype("float32")
    ids = np.array([c["id"] for c in chunks]).astype("int64")
    normalized = cosine_to_faiss(embeddings)

    index = create_faiss_index(normalized.shape[1])
    index.add_with_ids(normalized, ids)

    save_index(index)
    logger.info("FAISS index rebuilt with %d vectors.", len(ids))
    return index


def ensure_index():
    index = load_index()
    if index is None:
        index = rebuild_faiss_index()
    return index

def embed_texts(texts):
    if SKIP_EMBEDDINGS:
        logger.warning("Returning fake embeddings for test mode")
        return np.random.rand(len(texts), 768).astype("float32")  # or your model’s dim
    logger.info("Embedding %d texts", len(texts))
    return model.encode(texts, normalize_embeddings=True).astype("float32")

def query_faiss(query, top_k=5):
    logger.info("Querying FAISS for: %s", query)
    index = ensure_index()
    if index is None:
        logger.error("No FAISS index available.")
        return []

    embedding = embed_texts([query])
    scores, ids = index.search(embedding, top_k)

    results = []
    for score, id_ in zip(scores[0], ids[0]):
        if id_ == -1:
            continue
        chunk = get_chunk_by_ids([id_])[0]
        results.append({"score": float(score), "chunk": chunk})
    return results


def clear_faiss_index():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
        logger.info("FAISS index file deleted: %s", INDEX_PATH)
    else:
        logger.warning("No FAISS index file to delete.")
