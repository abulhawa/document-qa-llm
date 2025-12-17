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

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest


def test_index_viewer_shows_size(monkeypatch):
    files = [
        {
            "filename": "a.txt",
            "path": "/a.txt",
            "filetype": "txt",
            "modified_at": "1",
            "created_at": "1",
            "indexed_at": "1",
            "num_chunks": 1,
            "qdrant_count": 0,
            "checksum": "c1",
            "bytes": 1024,
        }
    ]
    monkeypatch.setattr(
        "utils.opensearch_utils.list_files_from_opensearch", lambda: files
    )

    at = AppTest.from_file("pages/2_index_viewer.py", default_timeout=10)
    at.run()

    df = at.dataframe[0].value
    assert "Size" in df.columns
    assert df["Size"].iloc[0] == 1024
