from celery.exceptions import Ignore
from .job_control import get_state


def ensure_job_can_start(job_id: str, task=None) -> None:
    state = get_state(job_id)
    if state in (None, "running"):
        return
    if state in {"pausing", "paused", "stopping"}:
        if task is not None:
            raise task.retry(countdown=30)
        raise RuntimeError("Job cannot start")
    if state == "canceled":
        raise Ignore()
