import pytest
from streamlit.testing.v1 import AppTest


def test_duplicate_table_renders(monkeypatch):
    checksums = ["c1", "c2"]
    file_template = {
        "filetype": "txt",
        "created_at": "1",
        "modified_at": "1",
        "indexed_at": "1",
        "num_chunks": 1,
    }
    files = {
        "c1": [
            {**file_template, "path": "a"},
            {**file_template, "path": "b"},
        ],
        "c2": [
            {**file_template, "path": "c"},
        ],
    }

    monkeypatch.setattr(
        "utils.opensearch_utils.get_duplicate_checksums", lambda: checksums
    )
    monkeypatch.setattr(
        "utils.opensearch_utils.get_files_by_checksum",
        lambda checksum: files[checksum],
    )

    at = AppTest.from_file("pages/3_duplicates_viewer.py", default_timeout=10)
    at.run()

    # One dataframe showing all duplicate files
    assert len(at.dataframe) == 1
    df = at.dataframe[0].value
    assert len(df) == 3
    assert df["Checksum"].value_counts().to_dict() == {"c1": 2, "c2": 1}
