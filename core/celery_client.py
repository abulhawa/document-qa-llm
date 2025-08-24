import os
from functools import lru_cache
from celery import Celery
from config import logger


@lru_cache(maxsize=1)
def get_celery() -> Celery:
    """Return a Celery client configured like the worker.

    Uses environment variables CELERY_BROKER_URL and CELERY_RESULT_BACKEND,
    matching worker defaults. Cached to avoid re-creating connections.
    """
    broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    app = Celery("document_qa_worker", broker=broker, backend=backend)
    logger.info("Celery client broker=%s backend=%s", broker, backend)
    return app
