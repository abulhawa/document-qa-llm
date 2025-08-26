import os
from worker.celery_worker import app as celery_app
from celery.signals import task_prerun, task_postrun, task_failure
from worker.audit import log_task


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
            return dst + hp[len(src) :]
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
def ingest_document(host_path: str, mode: str = "reingest") -> dict:
    """
    mode:
      - "reingest": force + replace (delete old OS/Qdrant entries first)
      - "reembed": force + no replace (re-process & re-embed without pre-deleting)
    """
    from core.ingestion import ingest_one  # heavy imports stay worker-side

    container_path = host_to_container_path(host_path)
    return ingest_one(host_path, container_path=container_path, force=True, replace=(mode=="reingest"), op=mode, source="celery")

