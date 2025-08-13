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


def test_ingest_validation_on_empty_selection(monkeypatch):
    monkeypatch.setattr("ui.ingestion_ui.run_file_picker", lambda: [])
    monkeypatch.setattr("ui.ingestion_ui.run_folder_picker", lambda: [])

    at = AppTest.from_file("pages/1_ingest.py", default_timeout=10)
    at.run()

    # Click the Ingest button with no selection
    at.button[2].click().run()

    assert at.warning[0].value == "Please select at least one file or folder."


def test_ingest_success_and_progress(monkeypatch):
    # Mock selection and ingestion
    monkeypatch.setattr("ui.ingestion_ui.run_file_picker", lambda: ["/tmp/a.txt"])
    monkeypatch.setattr("ui.ingestion_ui.run_folder_picker", lambda: [])

    progress_updates = []

    def fake_ingest(paths, progress_callback):
        progress_callback(0, 2, 0)
        progress_updates.append(0)
        progress_callback(1, 2, 0)
        progress_updates.append(0.5)
        progress_callback(2, 2, 0)
        progress_updates.append(1)
        return [
            {
                "path": paths[0],
                "success": True,
                "status": "ok",
                "num_chunks": 1,
                "batches": 1,
            }
        ]

    monkeypatch.setattr("core.ingestion.ingest", fake_ingest)

    at = AppTest.from_file("pages/1_ingest.py", default_timeout=10)
    at.run()

    # Select file and ingest
    at.button[0].click().run()
    at.button[2].click().run()

    # Success notice and log row
    assert "Queued" in at.success[1].value
    assert len(at.dataframe[0].value) == 1

    # Progress callback invoked for 0 -> 50 -> 100%
    assert progress_updates == [0, 0.5, 1]

    def find_progress(node):
        elems = []
        if getattr(node, "type", "") == "progress":
            elems.append(node)
        for child in getattr(node, "children", {}).values() if hasattr(node, "children") else []:
            elems.extend(find_progress(child))
        return elems

    progress_elems = find_progress(at._tree[0])
    assert progress_elems[0].value == 100
