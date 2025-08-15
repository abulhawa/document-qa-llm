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
