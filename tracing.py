from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
import importlib
import importlib.util
from typing import Any, Iterator, cast

_HAS_TRACING_DEPS = all(
    importlib.util.find_spec(module) is not None
    for module in (
        "phoenix.otel",
        "opentelemetry.trace",
        "openinference.semconv.trace",
    )
)

if _HAS_TRACING_DEPS:
    register = importlib.import_module("phoenix.otel").register
    opentelemetry_trace = importlib.import_module("opentelemetry.trace")
    openinference_trace = importlib.import_module("openinference.semconv.trace")
    get_current_span = cast(
        Any,
        getattr(opentelemetry_trace, "get_current_span"),
    )
    Status = cast(Any, getattr(opentelemetry_trace, "Status"))
    StatusCode = cast(Any, getattr(opentelemetry_trace, "StatusCode"))
    SpanAttributes = cast(Any, getattr(openinference_trace, "SpanAttributes"))
    OpenInferenceSpanKindValues = cast(
        Any, getattr(openinference_trace, "OpenInferenceSpanKindValues")
    )
else:

    class StatusCode(Enum):
        UNSET = 0
        OK = 1
        ERROR = 2

    class Status:
        def __init__(self, status_code: StatusCode, description: str | None = None):
            self.status_code = status_code
            self.description = description

    class SpanAttributes:
        INPUT_VALUE = "input"
        OUTPUT_VALUE = "output"
        OPENINFERENCE_SPAN_KIND = "openinference.span.kind"

    class OpenInferenceSpanKindValues(Enum):
        CHAIN = "chain"
        LLM = "llm"
        RETRIEVER = "retriever"
        EMBEDDING = "embedding"
        TOOL = "tool"

    class _DummySpan:
        def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def record_exception(self, _error: Exception) -> None:
            return None

        def set_status(self, _status: Status) -> None:
            return None

    class _DummyTracer:
        @contextmanager
        def start_as_current_span(self, _name: str) -> Iterator[_DummySpan]:
            yield _DummySpan()

    def get_current_span() -> _DummySpan:
        return _DummySpan()

    def register(*_args: Any, **_kwargs: Any):
        class _DummyProvider:
            def get_tracer(self, _name: str) -> _DummyTracer:
                return _DummyTracer()

        return _DummyProvider()


# Register tracer provider with Phoenix (singleton)
_tracer_provider = register(
    project_name="LocalDocQA",
    auto_instrument=False,
    set_global_tracer_provider=True,
    batch=True,
)

tracer = _tracer_provider.get_tracer("LocalDocQA")


# OpenInference span kinds
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
LLM = OpenInferenceSpanKindValues.LLM.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
STATUS_OK = StatusCode.OK

# OpenInference attribute keys
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE


@contextmanager
def start_span(name: str, kind: str):
    """Start a span with the appropriate OpenInference kind attribute."""
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind)
        yield span


def record_span_error(span, error: Exception):
    """Attach error information to a span and mark it as failed."""
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


__all__ = [
    "CHAIN",
    "EMBEDDING",
    "INPUT_VALUE",
    "LLM",
    "OUTPUT_VALUE",
    "RETRIEVER",
    "STATUS_OK",
    "TOOL",
    "Status",
    "StatusCode",
    "get_current_span",
    "record_span_error",
    "start_span",
    "tracer",
]
