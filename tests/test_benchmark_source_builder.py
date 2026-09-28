from pathlib import Path

from scripts.build_benchmark_source import _safe_name, _split_source_files


def test_split_source_files_handles_officeqa_semicolon_format():
    assert _split_source_files("a.txt; b.txt ;") == ["a.txt", "b.txt"]


def test_split_source_files_accepts_existing_sequence():
    assert _split_source_files(["a.txt", "b.txt"]) == ["a.txt", "b.txt"]


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
