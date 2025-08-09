from celery import Celery
import os

# Connect to Redis (host will be the docker-compose service name)
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
app = Celery("document_qa_worker", broker=broker_url)
app.config_from_object("config")

# Autodiscover tasks from your main project
app.autodiscover_tasks(["core"], related_name="embedding_tasks")
app.autodiscover_tasks(["core"], related_name="ingestion_tasks")
