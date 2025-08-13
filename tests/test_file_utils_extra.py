import os
from utils.file_utils import (
    compute_checksum,
    normalize_path,
    hash_path,
    get_file_size,
    get_file_timestamps,
)


def test_file_utils(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("content")

    checksum = compute_checksum(str(file_path))
    assert len(checksum) == 64

    assert normalize_path("a\\b/c") == "a/b/c"

    assert hash_path("abc")

    assert get_file_size("does-not-exist") == 0
    ts = get_file_timestamps("does-not-exist")
    assert ts == {"created": "", "modified": ""}
