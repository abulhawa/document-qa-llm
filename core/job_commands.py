from .celery_client import get_celery
from .job_control import set_state, pop_all_tasks, add_task, incr_stat
from .job_queue import (
    pop_all_retry,
    pop_all_active,
    add_retry,
)


def pause_job(job_id: str) -> None:
    set_state(job_id, "pausing")
    task_ids = pop_all_tasks(job_id)
    if task_ids:
        get_celery().control.revoke(task_ids, terminate=False)
    set_state(job_id, "paused")


def resume_job(job_id: str) -> None:
    app = get_celery()
    for path in pop_all_retry(job_id):
        result = app.send_task(
            "core.tasks.ingest_file_task", kwargs={"job_id": job_id, "path": path}
        )
        add_task(job_id, result.id)
        incr_stat(job_id, "enqueued", 1)
    set_state(job_id, "running")


def cancel_job(job_id: str) -> None:
    set_state(job_id, "canceled")
    task_ids = pop_all_tasks(job_id)
    if task_ids:
        get_celery().control.revoke(task_ids, terminate=True, signal="SIGKILL")
    for path in pop_all_active(job_id):
        add_retry(job_id, path)


def stop_job(job_id: str) -> None:
    set_state(job_id, "stopping")
