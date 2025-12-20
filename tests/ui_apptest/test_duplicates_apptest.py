import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest


def test_duplicate_table_renders(monkeypatch):
    checksums = ["c1", "c2"]
    file_template = {
        "filetype": "txt",
        "created_at": "1",
        "modified_at": "1",
        "indexed_at": "1",
        "num_chunks": 1,
        "bytes": 100,
        "canonical_path": "a",
    }
    files = {
        "c1": [
            {**file_template, "path": "a", "location_type": "canonical"},
            {**file_template, "path": "b", "location_type": "alias"},
        ],
        "c2": [
            {**file_template, "path": "c", "location_type": "canonical", "canonical_path": "c"},
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
    assert set(df["Location Type"].unique()) == {"Canonical", "Alias"}
    assert "Size" in df.columns


def test_duplicate_alias_locations_surface(monkeypatch):
    checksums = ["dup"]
    base = {
        "filetype": "txt",
        "created_at": "1",
        "modified_at": "1",
        "indexed_at": "1",
        "num_chunks": 3,
        "bytes": 2048,
        "canonical_path": "/docs/main.txt",
    }
    files = {
        "dup": [
            {**base, "path": "/docs/main.txt", "location_type": "canonical"},
            {**base, "path": "/alias/first.txt", "location_type": "alias"},
            {**base, "path": "/alias/second.txt", "location_type": "alias"},
        ]
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

    df = at.dataframe[0].value
    assert set(df["Location"]) == {
        "/docs/main.txt",
        "/alias/first.txt",
        "/alias/second.txt",
    }
    alias_rows = df[df["Location Type"] == "Alias"]
    assert alias_rows["Canonical Path"].unique().tolist() == ["/docs/main.txt"]
