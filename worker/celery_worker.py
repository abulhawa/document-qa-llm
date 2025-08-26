import os
from celery import Celery

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
    include=["worker.tasks", "core.ingestion_tasks"],
)
