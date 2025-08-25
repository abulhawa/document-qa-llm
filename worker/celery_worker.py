from celery import Celery
import os

# Connect to Redis (host will be the docker-compose service name)
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
backend_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
queue = os.getenv("CELERY_TASK_QUEUE", "celery")
app = Celery("document_qa_worker", broker=broker_url, backend=backend_url)
app.conf.task_default_queue = queue
app.config_from_object("config")

app.autodiscover_tasks(["core"])
# Autodiscover tasks from your main project
app.autodiscover_tasks(["core"], related_name="ingestion_tasks")
