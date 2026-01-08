from typing import Any, Callable

class _Signal:
    def connect(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Any]: ...

worker_ready = _Signal()
task_prerun = _Signal()
task_postrun = _Signal()
task_failure = _Signal()
