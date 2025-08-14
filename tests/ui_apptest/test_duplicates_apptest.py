import pytest
from streamlit.testing.v1 import AppTest


def test_duplicate_groups_render_and_delete(monkeypatch):
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

    subheaders = [s.value for s in at.subheader]
    assert "c1 (2 files)" in subheaders
    assert "c2 (1 file)" in subheaders

    delete_buttons = [b for b in at.button if b.label == "Delete group"]
    assert len(delete_buttons) == 2

    delete_buttons[0].click().run()

    assert [s.value for s in at.subheader] == ["c2 (1 file)"]
    assert len(at.session_state["duplicate_groups"]) == 1
    assert at.session_state["duplicate_groups"][0]["checksum"] == "c2"
