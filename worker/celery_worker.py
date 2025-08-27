import os
from celery import Celery
from celery.signals import worker_ready


app = Celery(
    "docqa",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1"),
)

app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": 3600},
    task_time_limit=1800,
    task_soft_time_limit=1500,
    task_default_queue="ingest",
    include=["worker.tasks"],
)


@worker_ready.connect
def _warmup(**_):
    from utils.opensearch_utils import (
        ensure_index_exists,
        ensure_fulltext_index_exists,
        ensure_ingest_log_index_exists,
    )
    from utils.qdrant_utils import ensure_collection_exists

    ensure_index_exists()
    ensure_fulltext_index_exists()
    ensure_ingest_log_index_exists()
    ensure_collection_exists()
