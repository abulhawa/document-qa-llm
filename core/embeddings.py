import os
from config import EMBEDDING_API_URL, EMBEDDING_BATCH_SIZE, logger
from typing import List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_session.mount(
    "http://",
    HTTPAdapter(
        pool_connections=16,
        pool_maxsize=16,
        max_retries=Retry(connect=3, read=0, total=0, backoff_factor=0.5),
    ),
)
_session.mount("https://", HTTPAdapter())

def embed_texts(texts: List[str], batch_size: int | None = None) -> List[List[float]]:
    bs = batch_size or EMBEDDING_BATCH_SIZE
    logger.info(
        f"Embedding API: batch_size={bs} total_texts={len(texts)} total_chars={sum(len(t) for t in texts)}"
    )
    resp = _session.post(
        EMBEDDING_API_URL,
        json={"texts": texts, "batch_size": bs},
        timeout=(3, 120),
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"]
