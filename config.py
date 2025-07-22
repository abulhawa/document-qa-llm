import os
import logging
from logging import Logger

# ───────────────────────────────────────
# 🔠 Embedding & Chunking
# ───────────────────────────────────────
EMBEDDING_API_URL = "http://localhost:8000/embed"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_SIZE = 768
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# ───────────────────────────────────────
# 🔍 Qdrant Vector Store
# ───────────────────────────────────────
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "document_chunks"
CHUNK_SCORE_THRESHOLD = 0.75


# ───────────────────────────────────────
# 🧠 LLM API (text-generation-webui)
# ───────────────────────────────────────
LLM_GENERATE_ENDPOINT = "http://localhost:5000/api/v1/generate"
LLM_COMPLETION_ENDPOINT = "http://localhost:5000/v1/completions"
LLM_CHAT_ENDPOINT = "http://localhost:5000/v1/chat/completions"
LLM_MODEL_LIST_ENDPOINT = "http://localhost:5000/v1/internal/model/list"
LLM_MODEL_LOAD_ENDPOINT = "http://localhost:5000/v1/internal/model/load"

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
