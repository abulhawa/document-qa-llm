import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Callable

# Ensure project root is on sys.path for direct test runs
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_configure(config):
    """Set safe default env for all tests so data isn't clobbered.

    - When running with -m e2e, set TEST_MODE=e2e so e2e tests run.
    - Otherwise, set TEST_MODE=test so e2e tests are skipped.
    - Always default NAMESPACE to 'test' if not provided, isolating indices.
    """
    try:
        markexpr = config.getoption("-m") or ""
    except Exception:
        markexpr = ""

    if "e2e" in markexpr:
        os.environ.setdefault("TEST_MODE", "e2e")
    else:
        os.environ.setdefault("TEST_MODE", "test")
    os.environ.setdefault("NAMESPACE", "test")


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


class _PhoenixOtelModule(ModuleType):
    register: Callable[..., _DummyProvider]


class _PhoenixModule(ModuleType):
    otel: _PhoenixOtelModule


phoenix = _PhoenixModule("phoenix")
otel = _PhoenixOtelModule("phoenix.otel")
otel.register = _register
phoenix.otel = otel
sys.modules.setdefault("phoenix", phoenix)
sys.modules.setdefault("phoenix.otel", otel)
