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
    alt_path = str(file_path).replace(os.sep, "\\")
    assert compute_checksum(alt_path) == checksum

    assert normalize_path("a\\b/c") == "a/b/c"

    assert hash_path("abc")

    assert get_file_size("does-not-exist") == 0
    assert get_file_size("does\\not\\exist") == 0

    # existing file returns ISO formatted timestamps
    ts_real = get_file_timestamps(str(file_path))
    assert ts_real["created"] and ts_real["modified"]
    from datetime import datetime

    datetime.fromisoformat(ts_real["created"])
    datetime.fromisoformat(ts_real["modified"])

    # missing file returns empty strings
    ts = get_file_timestamps("does-not-exist")
    assert ts == {"created": "", "modified": ""}
    ts = get_file_timestamps("does\\not\\exist")
    assert ts == {"created": "", "modified": ""}
