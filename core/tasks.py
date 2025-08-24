from typing import Dict, Any

from celery import shared_task

from core.paths import to_worker_path
from utils.file_utils import compute_checksum as file_checksum
from core.file_loader import load_documents as load_with_langchain


# Placeholder ensure_job_can_start; in a real system this would check job state.
def ensure_job_can_start(job_id: str, task=None) -> None:
    pass


@shared_task(bind=True, acks_late=True, autoretry_for=(), retry_backoff=True, max_retries=None)
def ingest_file_task(self, *, job_id: str, path: str) -> Dict[str, Any]:
    ensure_job_can_start(job_id, task=self)
    real_path = to_worker_path(path)
    checksum = file_checksum(real_path)
    docs = load_with_langchain(real_path)
    return {"job_id": job_id, "path": real_path, "checksum": checksum, "docs": docs}
