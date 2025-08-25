from core.job_queue import (
    push_pending,
    add_active,
    add_retry,
    list_jobs,
    clear_job,
    pending_count,
    active_count,
    retry_count,
)
from core.job_control import add_task, pop_all_tasks


def test_list_jobs_and_clear_job(monkeypatch):
    class StubControl:
        def __init__(self):
            self.revoked = []

        def revoke(self, task_id, terminate=False):
            self.revoked.append((task_id, terminate))

    class StubApp:
        def __init__(self):
            self.control = StubControl()

    stub_app = StubApp()
    monkeypatch.setattr("core.celery_client.get_celery", lambda: stub_app)

    job = "jobtest"
    push_pending(job, "p")
    add_active(job, "a")
    add_retry(job, "r")
    add_task(job, "tid1")

    assert job in list_jobs()

    counts = clear_job(job)
    assert counts == {
        "pending": 1,
        "active": 1,
        "active_started": 1,
        "needs_retry": 1,
        "tasks": 1,
    }
    assert pending_count(job) == 0
    assert active_count(job) == 0
    assert retry_count(job) == 0
    assert pop_all_tasks(job) == []
    assert stub_app.control.revoked == [("tid1", False)]
    assert job not in list_jobs()
