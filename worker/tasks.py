import os
from worker.celery_worker import app as celery_app

def host_to_container_path(host_path: str) -> str:
    # Uses DOC_PATH_MAP like 'C:/=>/host-c;G:/=>/host-g'
    mapping = []
    env = os.getenv("DOC_PATH_MAP", "")
    for part in env.split(";"):
        if "=>" in part:
            src, dst = part.split("=>", 1)
            mapping.append((src.rstrip("/\\").lower(), dst.rstrip("/")))
    mapping.sort(key=lambda p: len(p[0]), reverse=True)
    hp = host_path.replace("\\", "/")
    low = hp.lower()
    for src, dst in mapping:
        if low.startswith(src):
            return dst + hp[len(src):]
    return hp

# Ensure worker also loads batch tasks (referenced by ingestion.py on large files)
import core.ingestion_tasks  # registers "core.ingestion_tasks.index_and_embed_chunks" if present

@celery_app.task(
    name="tasks.ingest_document",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 5},
)
def ingest_document(host_path: str) -> dict:
    from core.ingestion import ingest_one  # heavy imports stay worker-side
    container_path = host_to_container_path(host_path)
    return ingest_one(container_path, source="celery")