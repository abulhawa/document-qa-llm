from typing import List, Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import EMBEDDING_API_URL, EMBEDDING_BATCH_SIZE, logger

EmbeddingInputType = Literal["passage", "query"]

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


def embed_texts(
    texts: List[str],
    batch_size: int | None = None,
    *,
    input_type: EmbeddingInputType = "passage",
) -> List[List[float]]:
    """Embed texts using an explicit retrieval role.

    Passage is the default for ingestion/document-side callers. Retrieval
    query callers must request input_type="query".
    """

    if input_type not in {"passage", "query"}:
        raise ValueError(f"unsupported embedding input_type: {input_type}")
    bs = batch_size or EMBEDDING_BATCH_SIZE
    logger.info(
        "Embedding API: input_type=%s batch_size=%s total_texts=%s total_chars=%s",
        input_type,
        bs,
        len(texts),
        sum(len(t) for t in texts),
    )
    resp = _session.post(
        EMBEDDING_API_URL,
        json={"texts": texts, "batch_size": bs, "input_type": input_type},
        timeout=(3, 120),
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"]
