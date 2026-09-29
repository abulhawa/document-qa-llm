import os

_DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
_DEFAULT_E5_REVISION = "d13f1b27baf31030b7fd040960d60d909913633f"

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", _DEFAULT_EMBEDDING_MODEL)
_revision_default = _DEFAULT_E5_REVISION if EMBEDDING_MODEL_NAME == _DEFAULT_EMBEDDING_MODEL else ""
EMBEDDING_MODEL_REVISION = os.getenv("EMBEDDING_MODEL_REVISION", _revision_default).strip()
_format_default = "e5" if "e5" in EMBEDDING_MODEL_NAME.lower() else "raw"
EMBEDDING_INPUT_FORMAT = os.getenv("EMBEDDING_INPUT_FORMAT", _format_default).strip().lower()
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
RERANK_MODEL_NAME = os.getenv(
    "RERANK_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANK_TOP_N_DEFAULT = int(os.getenv("RERANK_TOP_N_DEFAULT", "5"))
