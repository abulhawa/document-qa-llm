import os
import logging
from logging import Logger

# ───────────────────────────────────────
# 📁 Directories
# ───────────────────────────────────────
TEMP_DIR: str = "temp_docs"
os.makedirs(TEMP_DIR, exist_ok=True)

# ───────────────────────────────────────
# 📄 File paths
# ───────────────────────────────────────
DOCS_FOLDER: str = "docs"  # optional if you keep a static folder for ingestion

# ───────────────────────────────────────
# 🔠 Embedding & Chunking
# ───────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-base"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
EMBEDDING_API_URL = "http://localhost:8000/embed"


# ───────────────────────────────────────
# 🔍 Qdrant Vector Store
# ───────────────────────────────────────
QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333
QDRANT_COLLECTION_NAME: str = "document_chunks"
CHUNK_SCORE_THRESHOLD: float = 0.75


# ───────────────────────────────────────
# 🧠 LLM API (text-generation-webui)
# ───────────────────────────────────────
LLM_GENERATE_ENDPOINT: str = "http://localhost:5000/api/v1/generate"
LLM_COMPLETION_ENDPOINT: str = "http://localhost:5000/v1/completions"
LLM_MODEL_LIST_ENDPOINT: str = "http://localhost:5000/v1/internal/model/list"
LLM_MODEL_LOAD_ENDPOINT: str = "http://localhost:5000/v1/internal/model/load"

# ───────────────────────────────────────
# 📋 Logging
# ───────────────────────────────────────
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"

def setup_logging() -> Logger:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    return logging.getLogger("document_qa")

logger: Logger = setup_logging()
