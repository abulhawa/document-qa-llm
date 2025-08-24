import types
import sys
from contextlib import contextmanager

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _stub_tracing(monkeypatch):
    """Provide a no-op tracing module to avoid external deps during tests."""

    class _Span:
        def set_attribute(self, *args, **kwargs):
            pass

        def set_status(self, *args, **kwargs):
            pass

    @contextmanager
    def start_span(name: str, kind: str):
        yield _Span()

    dummy = types.SimpleNamespace(
        start_span=start_span,
        CHAIN="CHAIN",
        INPUT_VALUE="INPUT",
        OUTPUT_VALUE="OUTPUT",
        STATUS_OK="OK",
        LLM="LLM",
        RETRIEVER="RETRIEVER",
        EMBEDDING="EMBEDDING",
        TOOL="TOOL",
    )
    monkeypatch.setitem(sys.modules, "tracing", dummy)


def test_ingest_enqueues(monkeypatch):
    monkeypatch.setattr("ui.ingestion_ui.run_file_picker", lambda: ["/tmp/a.txt"])
    monkeypatch.setattr("ui.ingestion_ui.run_folder_picker", lambda: [])
    monkeypatch.setattr("utils.opensearch_utils.missing_indices", lambda: [])

    calls = []

    def fake_enqueue(paths, **kwargs):
        calls.append(list(paths))
        return {"job_id": "default", "enqueued": len(list(paths))}

    monkeypatch.setattr("ui.ingest_client.enqueue_ingest", fake_enqueue)

    at = AppTest.from_file("pages/1_ingest.py", default_timeout=10)
    at.run()

    at.button[0].click().run()

    assert calls == [["/tmp/a.txt"]]
    assert "Queued 1 file(s) for ingestion." in at.success[1].value
    assert len(at.dataframe[0].value) == 1
