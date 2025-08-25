"""Lightweight ingestion client used by Streamlit pages.
Queues files for ingestion and reports job statistics without importing
heavy worker-only modules.
"""
from __future__ import annotations

import os
from typing import Iterable, Dict, Any

import redis

from config import logger
from core.celery_client import get_celery
from core.job_queue import active_count, retry_count, celery_queue_len
from core.job_control import (
    set_state,
    get_state,
    incr_stat,
    get_stats,
    add_task,
)
from core.discovery_filters import should_skip

DEFAULT_JOB_ID = "default"


def purge_queue(queue: str | None = None) -> int:
    """Purge tasks from the broker queue.

    If *queue* is ``None`` or ``"celery"`` the default Celery control purge is
    used. Otherwise we drain the named queue using Kombu to avoid shelling out.
    """

    app = get_celery()
    if queue in (None, "", "celery"):
        return app.control.purge()

    from kombu import Connection, Queue

    broker_url = os.getenv("CELERY_BROKER_URL")
    n = 0
    if not broker_url:
        return n
    with Connection(broker_url) as conn:
        chan = conn.channel()
        q = Queue(name=queue)
        while True:
            msg = chan.basic_get(q.name, no_ack=True)
            if not msg:
                break
            n += 1
    return n


def revoke_job_tasks(job_id: str, terminate: bool = False) -> int:
    """Revoke any Celery tasks associated with *job_id*.

    Returns the number of tasks revoked. The Redis set is left intact so callers
    may clear it separately if desired.
    """

    try:
        r = redis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379/2"),
            decode_responses=True,
            socket_connect_timeout=0.1,
        )
    except Exception:
        return 0

    key = f"job:{job_id}:tasks"
    task_ids = list(r.smembers(key) or [])
    app = get_celery()
    for tid in task_ids:
        app.control.revoke(tid, terminate=terminate)
    return len(task_ids)

def enqueue_ingest(
    paths: Iterable[str],
    *,
    op: str = "ingest",
    force: bool = False,
    source: str = "ui",
    job_id: str = DEFAULT_JOB_ID,
) -> Dict[str, Any]:
    """Queue file paths for background ingestion.

    Args:
        paths: iterable of file paths to ingest.
        op: operation name, informational only.
        force: ignored flag kept for API compatibility.
        source: caller identifier, informational only.
        job_id: ingestion job identifier.

    Returns:
        Dict with counts of enqueued and skipped paths.
    """
    enqueued = 0
    skipped = 0
    for p in paths:
        if should_skip(p):
            skipped += 1
            continue
        result = get_celery().send_task(
            "core.tasks.ingest_file_task", kwargs={"job_id": job_id, "path": p}
        )
        add_task(job_id, result.id)
        incr_stat(job_id, "registered", 1)
        incr_stat(job_id, "enqueued", 1)
        enqueued += 1
    if enqueued:
        logger.info("Queued %s task(s); celery pending=%s", enqueued, celery_queue_len())
    if enqueued and get_state(job_id) != "running":
        set_state(job_id, "running")
    return {"job_id": job_id, "enqueued": enqueued, "skipped": skipped}

def job_stats(job: str = DEFAULT_JOB_ID) -> Dict[str, Any]:
    """Return current statistics for a job."""
    stats = get_stats(job)
    active = active_count(job)
    retry = retry_count(job)
    registered = stats.get("registered", 0)
    done = stats.get("done", 0)
    failed = stats.get("failed", 0)
    pending = max(registered - done - failed - active - retry, 0)
    return {
        "job_id": job,
        "state": get_state(job),
        "pending": pending,
        "active": active,
        "retry": retry,
        "stats": stats,
        "celery_queue": celery_queue_len(),
    }
