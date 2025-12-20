import os
from utils.file_utils import (
    compute_checksum,
    normalize_path,
    hash_path,
    get_file_size,
    get_file_timestamps,
    choose_canonical_path,
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
    ts = get_file_timestamps("does-not-exist")
    assert ts == {"created": "", "modified": ""}
    ts = get_file_timestamps("does\\not\\exist")
    assert ts == {"created": "", "modified": ""}


def test_choose_canonical_path_prefers_shallow_directory(tmp_path):
    deep_path = tmp_path / "level1" / "level2" / "file.txt"
    deep_path.parent.mkdir(parents=True)
    deep_path.write_text("x")
    shallow_path = tmp_path / "very_very_long_filename_but_shallow.txt"
    shallow_path.write_text("x")

    canonical = choose_canonical_path([str(deep_path), str(shallow_path)])
    assert canonical == str(shallow_path)
