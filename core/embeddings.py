from typing import List
import requests
import time
from config import EMBEDDING_API_URL, logger
# from tracing import get_tracer
from opentelemetry.trace import get_current_span


# tracer = get_tracer(__name__)

def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    span = get_current_span()
    span.set_attribute("embedding.num_inputs", len(texts))
    span.set_attribute("embedding.batch_size", batch_size)

    try:
        total_chars = sum(len(t) for t in texts)
        total_words = sum(len(t.split()) for t in texts)
        span.set_attribute("embedding.total_chars", total_chars)
        span.set_attribute("embedding.total_words", total_words)
        span.set_attribute("embedding.avg_chars_per_text", total_chars // len(texts))
        span.set_attribute("embedding.avg_words_per_text", total_words // len(texts))

        logger.info(f"Embedding {len(texts)} texts via API...")

        start_time = time.time()
        response = requests.post(
            EMBEDDING_API_URL,
            json={"texts": texts, "batch_size": batch_size},
            timeout=15,
        )
        response.raise_for_status()
        duration = time.time() - start_time

        span.set_attribute("embedding.api_duration_secs", round(duration, 3))

        embeddings = response.json()["embeddings"]
        if embeddings:
            span.set_attribute("embedding.vector_size", len(embeddings[0]))

        return embeddings

    except requests.RequestException as e:
        logger.error("Embedding API request failed: %s", str(e))
        raise RuntimeError(f"Embedding API error: {e}")

def embed_text(text: str) -> List[float]:
    span = get_current_span()
    span.set_attribute("embedding.single.length_chars", len(text))
    span.set_attribute("embedding.single.length_words", len(text.split()))

    try:
        logger.info("Embedding a single text via API...")
        start_time = time.time()

        response = requests.post(
            f"{EMBEDDING_API_URL}/embed",
            json={"text": text}
        )
        response.raise_for_status()
        duration = time.time() - start_time

        embedding = response.json()["embedding"]
        span.set_attribute("embedding.api_duration_secs", round(duration, 3))
        span.set_attribute("embedding.vector_size", len(embedding))

        return embedding

    except requests.RequestException as e:
        logger.error("Embedding API request failed: %s", str(e))
        raise RuntimeError(f"Embedding API error: {e}")