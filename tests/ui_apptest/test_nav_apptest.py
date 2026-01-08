import re
from pathlib import Path
import sys
import types
from typing import Any, cast

import pytest

# provide a minimal opensearchpy stub so streamlit pages can be imported
opensearchpy_stub = cast(Any, types.ModuleType("opensearchpy"))
opensearchpy_stub.exceptions = cast(Any, types.ModuleType("exceptions"))
opensearchpy_stub.exceptions.NotFoundError = Exception
opensearchpy_stub.exceptions.TransportError = Exception
opensearchpy_stub.OpenSearch = object
opensearchpy_stub.helpers = types.SimpleNamespace()
sys.modules.setdefault("opensearchpy", opensearchpy_stub)
sys.modules.setdefault("opensearchpy.exceptions", opensearchpy_stub.exceptions)

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

# Mapping of page script to expected title text and navigation label
PAGES = [
    ("pages/0_chat.py", "Ask Your Documents", "Ask Your Documents"),
    ("pages/1_ingest.py", "Ingest Documents", "Ingest Documents"),
    ("pages/8_storage_index.py", None, "Storage & Index"),
    ("pages/9_admin.py", None, "Admin"),
]

TITLE_PAGES = [
    ("pages/0_chat.py", "Ask Your Documents"),
    ("pages/1_ingest.py", "Ingest Documents"),
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

    # Assert nav link labels
    sidebar_links = [link for _, _, link in PAGES]
    assert sidebar_links == [
        "Ask Your Documents",
        "Ingest Documents",
        "Storage & Index",
        "Admin",
    ]

    # Navigate to select pages and verify title text
    for page_path, expected_title in TITLE_PAGES:
        at.switch_page(page_path).run()
        assert expected_title in at.title[0].value
