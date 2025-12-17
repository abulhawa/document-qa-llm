import os
from worker.celery_worker import app as celery_app
from celery.signals import task_prerun, task_postrun, task_failure
from worker.audit import log_task
from requests.exceptions import ReadTimeout, ConnectionError


@task_prerun.connect
def _audit_start(task_id=None, task=None, args=None, kwargs=None, **_):
    if task and getattr(task, "name", "").startswith(("tasks.", "core.")):
        log_task("start", task_id, task.name)


@task_postrun.connect
def _audit_done(task_id=None, task=None, retval=None, state=None, **_):
    if task and getattr(task, "name", "").startswith(("tasks.", "core.")):
        log_task("success", task_id, task.name, state=state, result=retval)


@task_failure.connect
def _audit_fail(task_id=None, exception=None, sender=None, **_):
    name = getattr(sender, "name", "unknown")
    log_task("failure", task_id, name, state="FAILURE", error=str(exception))


# --- tiny mapper: host (Windows) -> container path (e.g., /host-c) ---
def host_to_container_path(host_path: str) -> str:
    # Uses DOC_PATH_MAP like 'C:/=>/host-c;G:/=>/host-g'
    pairs = []
    env = os.getenv("DOC_PATH_MAP", "")
    for part in env.split(";"):
        if "=>" in part:
            src, dst = part.split("=>", 1)
            src = src.rstrip("/\\").lower()
            dst = dst.rstrip("/\\")
            pairs.append((src, dst))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    hp = host_path.replace("\\", "/")
    low = hp.lower()
    for src, dst in pairs:
        if low.startswith(src):
            return dst + hp[len(src) :]
    return hp


@celery_app.task(
    name="tasks.ingest_document",
    acks_late=True,
    queue="ingest",
    autoretry_for=(ReadTimeout, ConnectionError, RuntimeError),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
    retry_kwargs={"max_retries": 8},
)
def ingest_document(host_path: str, mode: str = "ingest") -> dict:
    """
    host_path: Windows host path selected in UI (stored in metadata)
    mode: 'ingest' (default), 'reembed', 'reingest'
    """
    from ingestion.orchestrator import ingest_one
    
    fs_path = host_to_container_path(host_path)
    if mode == "ingest":
        return ingest_one(
            host_path,
            fs_path=fs_path,
            force=False,
            replace=False,
            op="ingest",
            source="celery",
        )
    if mode == "reembed":
        return ingest_one(
            host_path,
            fs_path=fs_path,
            force=True,
            replace=False,
            op="reembed",
            source="celery",
        )
    return ingest_one(
        host_path,
        fs_path=fs_path,
        force=True,
        replace=True,
        op="reingest",
        source="celery",
    )


@celery_app.task(
    name="tasks.delete_document",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def delete_document(path: str) -> dict:
    """
    path: EXACT value stored in OpenSearch payloads (path.keyword). 
    We purge:
      1) Qdrant vectors by chunk IDs (fetched from OS),
      2) OS chunk docs (documents index),
      3) OS full-text doc(s) (full_text index).
    """
    from config import logger
    from utils.opensearch_utils import (
        get_chunk_ids_by_path,
        delete_chunks_by_path,
        delete_fulltext_by_path,
    )
    from utils.qdrant_utils import delete_vectors_by_ids

    ids = []
    try:
        ids = get_chunk_ids_by_path(path)
    except Exception as e:
        logger.exception("List chunk IDs failed for %s: %s", path, e)

    deleted_vec = 0
    try:
        deleted_vec = delete_vectors_by_ids(ids)
    except Exception as e:
        logger.exception("Qdrant delete failed for %s: %s", path, e)

    deleted_chunks = 0
    try:
        deleted_chunks = delete_chunks_by_path(path)
    except Exception as e:
        logger.exception("OS chunk delete failed for %s: %s", path, e)

    deleted_fulltext = 0
    try:
        deleted_fulltext = delete_fulltext_by_path(path)
    except Exception as e:
        logger.exception("OS fulltext delete failed for %s: %s", path, e)

    return {
        "success": True,
        "path": path,
        "deleted": {
            "qdrant_points": deleted_vec,
            "os_chunks": deleted_chunks,
            "os_fulltext": deleted_fulltext,
        },
    }
