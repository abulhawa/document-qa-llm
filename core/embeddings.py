from typing import List
import requests
from config import EMBEDDING_API_URL, logger
from tracing import get_current_span, record_span_error


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Send a batch of texts to the embedding API and return their embeddings.
    Tracing only captures metadata (e.g., input sizes, timing), not content or vectors.
    """
    span = get_current_span()
    span.set_attribute("num_inputs", len(texts))
    span.set_attribute("batch_size", batch_size)

    try:
        span.set_attribute("total_chars", sum(len(t) for t in texts))
        span.set_attribute("total_words", sum(len(t.split()) for t in texts))

        logger.info(f"Embedding {len(texts)} texts via API...")

        response = requests.post(
            EMBEDDING_API_URL,
            json={"texts": texts, "batch_size": batch_size},
            timeout=30,
        )
        response.raise_for_status()

        embeddings = response.json()["embeddings"]
        if embeddings:
            span.set_attribute("vector_size", len(embeddings[0]))

        return embeddings

    except requests.RequestException as e:
        logger.error("Embedding API request failed: %s", str(e))
        record_span_error(span, e)
        raise RuntimeError(f"Embedding API error: {e}")
