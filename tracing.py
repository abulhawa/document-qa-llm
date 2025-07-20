# tracing.py

from phoenix.otel import register
from opentelemetry import trace

# Register Phoenix tracer once
_tracer_provider = register(
    project_name="LocalDocQA",
    auto_instrument=True,
    batch=True
)

def get_tracer(name: str = "LocalDocQA"):
    return _tracer_provider.get_tracer(name)
