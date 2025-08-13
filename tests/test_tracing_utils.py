from tracing import record_span_error, StatusCode, Status


class DummySpan:
    def __init__(self):
        self.exc = None
        self.status = None

    def record_exception(self, exc):
        self.exc = exc

    def set_status(self, status):
        self.status = status


def test_record_span_error():
    span = DummySpan()
    err = ValueError("bad")
    record_span_error(span, err)
    assert span.exc is err
    assert isinstance(span.status, Status)
    assert span.status.status_code == StatusCode.ERROR
