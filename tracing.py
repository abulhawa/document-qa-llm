from phoenix.otel import register

_tracer_provider = None  # singleton guard


def get_tracer(name: str = __name__):
    global _tracer_provider
    if _tracer_provider is None:
        _tracer_provider = register(
            project_name="LocalDocQA",
            auto_instrument=False,
            set_global_tracer_provider=True,
            batch=True,
        )
    return _tracer_provider.get_tracer(name)
