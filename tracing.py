from contextlib import contextmanager

try:  # Prefer real tracing backends when available
    from phoenix.otel import register
    from opentelemetry import trace
    from opentelemetry.trace import get_current_span
    from openinference.semconv.trace import (
        SpanAttributes,
        OpenInferenceSpanKindValues,
    )
    from opentelemetry.trace import Status, StatusCode

    _tracer_provider = register(
        project_name="LocalDocQA",
        auto_instrument=False,
        set_global_tracer_provider=True,
        batch=True,
    )

    tracer = _tracer_provider.get_tracer("LocalDocQA")

except Exception:  # Fallback to no-op tracing when optional deps are missing
    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"

    class Status:
        def __init__(self, status_code, description: str | None = None):
            self.status_code = status_code
            self.description = description

    class _SpanAttributes:
        INPUT_VALUE = "openinference.input_value"
        OUTPUT_VALUE = "openinference.output_value"
        OPENINFERENCE_SPAN_KIND = "openinference.span.kind"

    class _OpenInferenceSpanKindValues:
        CHAIN = type("_V", (), {"value": "chain"})()
        LLM = type("_V", (), {"value": "llm"})()
        RETRIEVER = type("_V", (), {"value": "retriever"})()
        EMBEDDING = type("_V", (), {"value": "embedding"})()
        TOOL = type("_V", (), {"value": "tool"})()

    class _NoOpSpan:
        def set_attribute(self, *_args, **_kwargs):
            return None

        def record_exception(self, *_args, **_kwargs):
            return None

        def set_status(self, *_args, **_kwargs):
            return None

    class _NoOpTracer:
        @contextmanager
        def start_as_current_span(self, _name: str):
            yield _NoOpSpan()

    def get_current_span():  # type: ignore[override]
        return _NoOpSpan()

    tracer = _NoOpTracer()

    SpanAttributes = _SpanAttributes
    OpenInferenceSpanKindValues = _OpenInferenceSpanKindValues


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
