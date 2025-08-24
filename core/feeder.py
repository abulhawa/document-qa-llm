import os
from .job_queue import pop_pending, add_active, inflight
from .job_control import incr_stat, add_task

MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", "2"))


def feed_once(job_id: str) -> int:
    from .tasks import ingest_file_task

    started = 0
    while inflight(job_id) < MAX_INFLIGHT:
        path = pop_pending(job_id)
        if not path:
            break
        add_active(job_id, path)
        result = ingest_file_task.delay(job_id=job_id, path=path)
        add_task(job_id, result.id)
        incr_stat(job_id, "enqueued", 1)
        started += 1
    return started
