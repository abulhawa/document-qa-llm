import os

# Default model name and batch size
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
RERANK_MODEL_NAME = os.getenv(
    "RERANK_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANK_TOP_N_DEFAULT = int(os.getenv("RERANK_TOP_N_DEFAULT", "5"))
