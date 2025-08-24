"""Lightweight ingestion client used by Streamlit pages.
Queues files for ingestion and reports job statistics without importing
heavy worker-only modules.
"""
from __future__ import annotations

from typing import Iterable, Dict, Any

from core.job_queue import (
    push_pending,
    pending_count,
    active_count,
    retry_count,
)
from core.job_control import (
    set_state,
    get_state,
    incr_stat,
    get_stats,
)
from core.discovery_filters import should_skip

DEFAULT_JOB_ID = "default"

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
        push_pending(job_id, p)
        incr_stat(job_id, "registered", 1)
        enqueued += 1
    if enqueued and get_state(job_id) != "running":
        set_state(job_id, "running")
    return {"job_id": job_id, "enqueued": enqueued, "skipped": skipped}

def job_stats(job: str = DEFAULT_JOB_ID) -> Dict[str, Any]:
    """Return current statistics for a job."""
    return {
        "job_id": job,
        "state": get_state(job),
        "pending": pending_count(job),
        "active": active_count(job),
        "retry": retry_count(job),
        "stats": get_stats(job),
    }
