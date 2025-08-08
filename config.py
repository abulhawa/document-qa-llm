import os
import logging
from logging import Logger


# ───────────────────────────────────────
# ⚙️ Ingestion Concurrency & Limits
# ───────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


# New ingestion controls
INGEST_MAX_WORKERS = _env_int("INGEST_MAX_WORKERS", 8)
INGEST_IO_CONCURRENCY = _env_int("INGEST_IO_CONCURRENCY", INGEST_MAX_WORKERS)
INGEST_MAX_FAILURES = _env_int("INGEST_MAX_FAILURES", 10)

# Optional: tune delete batching / timeouts used by utils (only if you want central control)
OPENSEARCH_DELETE_BATCH = _env_int("OPENSEARCH_DELETE_BATCH", 1024)
QDRANT_DELETE_BATCH = _env_int("QDRANT_DELETE_BATCH", 64)
OPENSEARCH_REQUEST_TIMEOUT = _env_int("OPENSEARCH_REQUEST_TIMEOUT", 60)

# ───────────────────────────────────────
# 🔠 Embedding & Chunking
# ───────────────────────────────────────
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8000/embed")
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)
EMBEDDING_SIZE = 768
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# ───────────────────────────────────────
# 🔍 Qdrant Vector Store
# ───────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "document_chunks"
CHUNK_SCORE_THRESHOLD = 0.75

# ───────────────────────────────────────
# 🔍 OPENSEARCH
# ───────────────────────────────────────
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")

# ───────────────────────────────────────
# 🧠 LLM API (text-generation-webui)
# ───────────────────────────────────────
LLM_GENERATE_ENDPOINT = "http://localhost:5000/api/v1/generate"
LLM_COMPLETION_ENDPOINT = "http://localhost:5000/v1/completions"
LLM_CHAT_ENDPOINT = "http://localhost:5000/v1/chat/completions"
LLM_MODEL_LIST_ENDPOINT = "http://localhost:5000/v1/internal/model/list"
LLM_MODEL_LOAD_ENDPOINT = "http://localhost:5000/v1/internal/model/load"
LLM_MODEL_INFO_ENDPOINT = "http://localhost:5000/v1/internal/model/info"

# ───────────────────────────────────────
# 📋 Logging
# ───────────────────────────────────────
logger = logging.getLogger("ingestion")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(handler)
