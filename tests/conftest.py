import types
import sys
from contextlib import contextmanager


class _DummySpan:
    def set_attribute(self, *args, **kwargs):
        pass

    def record_exception(self, *args, **kwargs):
        pass

    def set_status(self, *args, **kwargs):
        pass


class _DummyTracer:
    @contextmanager
    def start_as_current_span(self, name):
        yield _DummySpan()


class _DummyProvider:
    def get_tracer(self, name):
        return _DummyTracer()


def _register(*args, **kwargs):
    return _DummyProvider()


phoenix = types.ModuleType("phoenix")
otel = types.ModuleType("phoenix.otel")
otel.register = _register
phoenix.otel = otel
sys.modules.setdefault("phoenix", phoenix)
sys.modules.setdefault("phoenix.otel", otel)

