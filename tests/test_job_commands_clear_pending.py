import core.job_commands as jc
from core.job_queue import push_pending, pending_len


class DummyCelery:
    def __init__(self):
        self.control = self
    def revoke(self, *args, **kwargs):
        pass
    def send_task(self, *args, **kwargs):
        class Result:
            id = "dummy"
        return Result()


def _setup_pending(job_id: str):
    push_pending(job_id, "a")
    push_pending(job_id, "b")
    assert pending_len(job_id) == 2


def test_cancel_job_clears_pending(monkeypatch):
    job_id = "cancel_pending"
    _setup_pending(job_id)
    monkeypatch.setattr(jc, "get_celery", lambda: DummyCelery())
    jc.cancel_job(job_id)
    assert pending_len(job_id) == 0


def test_pause_job_preserves_pending(monkeypatch):
    job_id = "pause_pending"
    _setup_pending(job_id)
    monkeypatch.setattr(jc, "get_celery", lambda: DummyCelery())
    jc.pause_job(job_id)
    assert pending_len(job_id) == 2


def test_resume_job_preserves_pending(monkeypatch):
    job_id = "resume_pending"
    _setup_pending(job_id)
    monkeypatch.setattr(jc, "get_celery", lambda: DummyCelery())
    jc.resume_job(job_id)
    assert pending_len(job_id) == 2


def test_stop_job_clears_pending(monkeypatch):
    job_id = "stop_pending"
    _setup_pending(job_id)
    monkeypatch.setattr(jc, "get_celery", lambda: DummyCelery())
    jc.stop_job(job_id)
    assert pending_len(job_id) == 0
