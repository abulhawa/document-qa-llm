import os
from functools import lru_cache
from celery import Celery

@lru_cache(maxsize=1)
def get_ui_celery() -> Celery:
    broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    app = Celery("docqa-ui", broker=broker, backend=backend)
    app.conf.update(
        task_default_queue="ingest",
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        broker_transport_options={"visibility_timeout": 3600},
        task_time_limit=1800,
        task_soft_time_limit=1500,
    )
    return app