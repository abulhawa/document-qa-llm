from phoenix.otel import register
from opentelemetry import trace
from opentelemetry.trace import get_current_span
from contextlib import contextmanager
from openinference.semconv.trace import (
    SpanAttributes,
    OpenInferenceSpanKindValues,
)
from opentelemetry.trace import Status, StatusCode


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
