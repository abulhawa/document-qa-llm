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


# Recycle worker processes to avoid memory creep during long runs
app.conf.worker_max_tasks_per_child = int(
    os.getenv("WORKER_MAX_TASKS_PER_CHILD", "200")
)

# Rate-limit heavy tasks cluster-wide (adjust via env if needed)
app.conf.task_annotations = {
    "tasks.ingest_document": {"rate_limit": os.getenv("INGEST_RATE_LIMIT", "24/m")},
    "tasks.delete_document": {"rate_limit": "60/m"},
}

app.conf.task_routes = {
    "tasks.ingest_document": {"queue": "ingest"},
}


@worker_ready.connect
def _warmup(**_):
    from utils.opensearch.indexes import ensure_index_exists
    from utils.qdrant_utils import ensure_collection_exists
    from config import CHUNKS_INDEX, FULLTEXT_INDEX, INGEST_LOG_INDEX

    ensure_index_exists(CHUNKS_INDEX)
    ensure_index_exists(FULLTEXT_INDEX)
    ensure_index_exists(INGEST_LOG_INDEX)
    ensure_collection_exists()
