# config.py
from __future__ import annotations
import logging, os

# Load .env files locally (safe no-ops in CI)
try:
    from dotenv import load_dotenv  # needs python-dotenv
    load_dotenv(".env.local", override=True)
    load_dotenv(".env")
except Exception:
    pass

def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

# ── Global mode & namespacing (for CI/e2e)
CI: bool = _env_bool("CI", False)
TEST_MODE: str = _env_str("TEST_MODE", "off")  # off|e2e|integration|ui_only
# Default namespace: 'ci' on CI, 'test' when TEST_MODE is active, otherwise empty
_default_ns = "ci" if CI else ("test" if TEST_MODE in ("e2e", "integration", "ui_only") else "")
NAMESPACE: str = _env_str("NAMESPACE", _default_ns)
INDEX_PREFIX: str = _env_str("INDEX_PREFIX", "")
_default_sha = os.getenv("GITHUB_SHA", "")[:7]
INDEX_SUFFIX: str = _env_str("INDEX_SUFFIX", f"sha{_default_sha}" if CI and _default_sha else "")

def _namespaced(base: str) -> str:
    parts = [INDEX_PREFIX, base, NAMESPACE, INDEX_SUFFIX]
    return "-".join([p for p in parts if p]).lower().replace("\\", "-").replace("/", "-") or base

# ── Ingestion concurrency & limits (keep your knobs)
INGEST_MAX_WORKERS         = _env_int("INGEST_MAX_WORKERS", 8)
INGEST_IO_CONCURRENCY      = _env_int("INGEST_IO_CONCURRENCY", INGEST_MAX_WORKERS)
INGEST_MAX_FAILURES        = _env_int("INGEST_MAX_FAILURES", 10)

# ── Optional batching / timeouts
OPENSEARCH_DELETE_BATCH    = _env_int("OPENSEARCH_DELETE_BATCH", 1024)
QDRANT_DELETE_BATCH        = _env_int("QDRANT_DELETE_BATCH", 64)
OPENSEARCH_REQUEST_TIMEOUT = _env_int("OPENSEARCH_REQUEST_TIMEOUT", 60)

# ── Embedding & chunking
EMBEDDING_API_URL    = _env_str("EMBEDDING_API_URL", "http://localhost:8000/embed")
EMBEDDING_MODEL_NAME = _env_str("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)
EMBEDDING_REQ_MAX_CHUNKS = int(os.getenv("EMBEDDING_REQ_MAX_CHUNKS", "64"))
EMBEDDING_SIZE       = _env_int("EMBEDDING_SIZE", 768)
CHUNK_SIZE           = _env_int("CHUNK_SIZE", 400)
CHUNK_OVERLAP        = _env_int("CHUNK_OVERLAP", 50)

# ── Qdrant
QDRANT_URL             = _env_str("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_BASE = _env_str("QDRANT_COLLECTION_BASE", "document_chunks")
QDRANT_COLLECTION      = _namespaced(QDRANT_COLLECTION_BASE)
QDRANT_FILE_VECTORS_COLLECTION_BASE = _env_str("QDRANT_FILE_VECTORS_COLLECTION_BASE", "file_vectors")
QDRANT_FILE_VECTORS_COLLECTION = _namespaced(QDRANT_FILE_VECTORS_COLLECTION_BASE)
CHUNK_SCORE_THRESHOLD  = float(_env_str("CHUNK_SCORE_THRESHOLD", "0.75"))

# ── OpenSearch
OPENSEARCH_URL         = _env_str("OPENSEARCH_URL", "http://localhost:9200")
CHUNKS_INDEX_BASE  = _env_str("CHUNKS_INDEX_BASE", "documents")
INGEST_LOG_INDEX_BASE  = _env_str("INGEST_LOG_INDEX_BASE", "ingest_logs")
CHUNKS_INDEX       = _namespaced(CHUNKS_INDEX_BASE)
INGEST_LOG_INDEX       = _namespaced(INGEST_LOG_INDEX_BASE)
FULLTEXT_INDEX_BASE = _env_str(
    "FULLTEXT_INDEX_BASE", "documents_full_text"
)
FULLTEXT_INDEX = _namespaced(FULLTEXT_INDEX_BASE)

# Inventory of files (per-path facts)
WATCH_INVENTORY_INDEX_BASE = _env_str("WATCH_INVENTORY_INDEX_BASE", "watch_inventory")
WATCH_INVENTORY_INDEX = _namespaced(WATCH_INVENTORY_INDEX_BASE)

# Watchlist of tracked prefixes/folders (configuration)
WATCHLIST_INDEX_BASE = _env_str("WATCHLIST_INDEX_BASE", "watchlists")
WATCHLIST_INDEX = _namespaced(WATCHLIST_INDEX_BASE)

# ── LLM API (text-generation-webui compatible)
LLM_BASE_URL            = _env_str("LLM_BASE_URL", "http://localhost:5000").rstrip("/")
LLM_GENERATE_ENDPOINT   = _env_str("LLM_GENERATE_ENDPOINT",   f"{LLM_BASE_URL}/api/v1/generate")
LLM_COMPLETION_ENDPOINT = _env_str("LLM_COMPLETION_ENDPOINT", f"{LLM_BASE_URL}/v1/completions")
LLM_CHAT_ENDPOINT       = _env_str("LLM_CHAT_ENDPOINT",       f"{LLM_BASE_URL}/v1/chat/completions")
LLM_MODEL_LIST_ENDPOINT = _env_str("LLM_MODEL_LIST_ENDPOINT", f"{LLM_BASE_URL}/v1/internal/model/list")
LLM_MODEL_LOAD_ENDPOINT = _env_str("LLM_MODEL_LOAD_ENDPOINT", f"{LLM_BASE_URL}/v1/internal/model/load")
LLM_MODEL_INFO_ENDPOINT = _env_str("LLM_MODEL_INFO_ENDPOINT", f"{LLM_BASE_URL}/v1/internal/model/info")

# ── CI toggles (handy for e2e)
USE_STUB_EMBEDDER = _env_bool("USE_STUB_EMBEDDER", CI or TEST_MODE in ("e2e", "ui_only"))
USE_STUB_LLM      = _env_bool("USE_STUB_LLM",      CI or TEST_MODE in ("e2e", "ui_only"))

# ── Logging
LOG_LEVEL = _env_str("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("ingestion")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)

def dump_config_for_debug() -> None:
    safe = {
        "CI": CI, "TEST_MODE": TEST_MODE, "NAMESPACE": NAMESPACE,
        "OPENSEARCH_URL": OPENSEARCH_URL, "CHUNKS_INDEX": CHUNKS_INDEX,
        "FULLTEXT_INDEX": FULLTEXT_INDEX,
        "WATCH_INVENTORY_INDEX": WATCH_INVENTORY_INDEX,
        "WATCHLIST_INDEX": WATCHLIST_INDEX,
        "INGEST_LOG_INDEX": INGEST_LOG_INDEX, "QDRANT_URL": QDRANT_URL,
        "QDRANT_COLLECTION": QDRANT_COLLECTION, "USE_STUB_EMBEDDER": USE_STUB_EMBEDDER,
        "USE_STUB_LLM": USE_STUB_LLM, "EMBEDDING_SIZE": EMBEDDING_SIZE,
    }
    logger.info("Effective config: %s", safe)
