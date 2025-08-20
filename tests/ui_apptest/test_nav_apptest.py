import re
from pathlib import Path
import sys
import types

import pytest

# provide a minimal opensearchpy stub so streamlit pages can be imported
opensearchpy_stub = types.ModuleType("opensearchpy")
opensearchpy_stub.exceptions = types.ModuleType("exceptions")
opensearchpy_stub.exceptions.NotFoundError = Exception
opensearchpy_stub.exceptions.TransportError = Exception
opensearchpy_stub.OpenSearch = object
opensearchpy_stub.helpers = types.SimpleNamespace()
sys.modules.setdefault("opensearchpy", opensearchpy_stub)
sys.modules.setdefault("opensearchpy.exceptions", opensearchpy_stub.exceptions)

from streamlit.testing.v1 import AppTest

# Mapping of page script to expected title text and navigation label
PAGES = [
    ("pages/0_chat.py", "\U0001F4AC Talk to Your Documents", "Chat"),
    ("pages/1_ingest.py", "\U0001F4E5 Ingest Documents", "Ingest"),
    ("pages/2_index_viewer.py", "\U0001F4C2 File Index Viewer", "Index Viewer"),
    ("pages/3_duplicates_viewer.py", "Duplicate Files", "Duplicates"),
    ("pages/4_ingest_logs.py", "\U0001F4DD Ingestion Logs", "Ingest Logs"),
]


@pytest.fixture(autouse=True)
def _mock_external_calls(monkeypatch):
    """Prevent network calls so AppTest can run offline."""
    monkeypatch.setattr(
        "core.llm.check_llm_status",
        lambda: {
            "active": False,
            "server_online": False,
            "model_loaded": False,
            "status_message": "",
            "current_model": None,
        },
    )
    monkeypatch.setattr(
        "utils.opensearch_utils.list_files_from_opensearch", lambda: []
    )
    monkeypatch.setattr(
        "utils.opensearch_utils.get_duplicate_checksums", lambda: []
    )
    monkeypatch.setattr(
        "utils.opensearch_utils.get_files_by_checksum", lambda checksum: []
    )
    monkeypatch.setattr(
        "utils.opensearch_utils.search_ingest_logs", lambda **kwargs: []
    )


def test_navigation_links_and_titles():
    at = AppTest.from_file("main.py")

    # Assert sidebar link labels
    sidebar_links = [link for _, _, link in PAGES]
    assert sidebar_links == [
        "Chat", "Ingest", "Index Viewer", "Duplicates", "Ingest Logs"
    ]

    # Navigate to each page and verify title text
    for page_path, expected_title, _ in PAGES:
        at.switch_page(page_path).run()
        assert at.title[0].value == expected_title
