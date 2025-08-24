from celery import current_app
from .job_control import set_state, pop_all_tasks
from .job_queue import (
    pop_all_retry,
    push_pending,
    pop_all_active,
    add_retry,
)


def pause_job(job_id: str) -> None:
    set_state(job_id, "pausing")
    task_ids = pop_all_tasks(job_id)
    if task_ids:
        current_app.control.revoke(task_ids, terminate=False)
    set_state(job_id, "paused")


def resume_job(job_id: str) -> None:
    for path in pop_all_retry(job_id):
        push_pending(job_id, path)
    set_state(job_id, "running")


def cancel_job(job_id: str) -> None:
    set_state(job_id, "canceled")
    task_ids = pop_all_tasks(job_id)
    if task_ids:
        current_app.control.revoke(task_ids, terminate=True, signal="SIGKILL")
    for path in pop_all_active(job_id):
        add_retry(job_id, path)


def stop_job(job_id: str) -> None:
    set_state(job_id, "stopping")
