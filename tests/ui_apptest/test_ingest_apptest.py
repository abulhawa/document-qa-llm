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


def test_ingest_success_and_progress(monkeypatch):
    # Mock selection and ingestion
    monkeypatch.setattr("ui.ingestion_ui.run_file_picker", lambda: ["/tmp/a.txt"])
    monkeypatch.setattr("ui.ingestion_ui.run_folder_picker", lambda: [])

    calls = {}

    def fake_enqueue(paths, mode="ingest"):
        calls["paths"] = paths
        return ["t1"]

    monkeypatch.setattr("ui.ingest_client.enqueue_paths", fake_enqueue)

    at = AppTest.from_file("pages/1_ingest.py", default_timeout=10)
    at.run()

    # Select file which triggers ingestion automatically
    at.button[0].click().run()

    # Success notice and log row
    assert "Found 1 path" in at.success[0].value
    assert "Queued 1 file(s) for ingestion." in at.info[0].value
    assert len(at.dataframe[0].value) == 1

    def find_progress(node):
        elems = []
        if getattr(node, "type", "") == "progress":
            elems.append(node)
        for child in (
            getattr(node, "children", {}).values() if hasattr(node, "children") else []
        ):
            elems.extend(find_progress(child))
        return elems

    progress_elems = find_progress(at._tree[0])
    assert progress_elems[0].value == 100
    assert calls["paths"] == ["/tmp/a.txt"]
