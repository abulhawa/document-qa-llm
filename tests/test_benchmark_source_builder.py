from pathlib import Path

from scripts.build_benchmark_source import (
    _safe_name,
    _select_hf_parquet_files,
    _split_source_files,
)


def test_split_source_files_handles_officeqa_semicolon_format():
    assert _split_source_files("a.txt; b.txt ;") == ["a.txt", "b.txt"]


def test_split_source_files_handles_officeqa_multiline_format():
    raw = (
        "treasury_bulletin_1939_07.txt\r\n"
        "treasury_bulletin_1939_08.txt\r\n"
        "treasury_bulletin_1939_09.txt\n"
        "treasury_bulletin_1939_10.txt"
    )
    assert _split_source_files(raw) == [
        "treasury_bulletin_1939_07.txt",
        "treasury_bulletin_1939_08.txt",
        "treasury_bulletin_1939_09.txt",
        "treasury_bulletin_1939_10.txt",
    ]


def test_split_source_files_accepts_existing_sequence():
    assert _split_source_files(["a.txt", "b.txt"]) == ["a.txt", "b.txt"]


def test_select_hf_parquet_files_selects_only_requested_split():
    files = [
        "data/test-00001-of-00002.parquet",
        "data/train-00000-of-00001.parquet",
        "data/test-00000-of-00002.parquet",
        "queries/test-00000-of-00001.parquet",
    ]
    assert _select_hf_parquet_files(files, prefix="data", split="test") == [
        "data/test-00000-of-00002.parquet",
        "data/test-00001-of-00002.parquet",
    ]


def test_safe_name_prevents_nested_paths():
    value = _safe_name("cs/9901001")
    assert "/" not in value
    assert "\\" not in value
    assert value


def test_pending_lock_file_is_valid_json():
    import json

    path = Path("evaluation/benchmarks/composite_v1.lock.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["benchmark_id"] == "composite-v1"
