from scripts.parse_benchmark_source import (
    _artifact_fingerprint,
    _normalise_metadata,
    _parser_fingerprint,
)


def test_normalise_metadata_rewrites_runner_source_path():
    metadata = {"source": "/home/runner/private/file.pdf", "page": 3}
    result = _normalise_metadata(metadata, "benchmark://composite-v1/officeqa/doc-1")
    assert result["source"] == "benchmark://composite-v1/officeqa/doc-1"
    assert result["page"] == 3


def test_parser_fingerprint_is_stable():
    first = _parser_fingerprint()
    second = _parser_fingerprint()
    assert first == second
    assert len(first) == 64


def test_artifact_fingerprint_includes_resolved_runtime():
    base = _artifact_fingerprint(
        "source-artifact",
        "source-lock",
        "parser-code",
        {"pypdf": "6.19.0", "fonttools": "4.58.0", "ftfy": "6.3.1"},
        "3.13.15",
    )
    package_change = _artifact_fingerprint(
        "source-artifact",
        "source-lock",
        "parser-code",
        {"pypdf": "6.19.1", "fonttools": "4.58.0", "ftfy": "6.3.1"},
        "3.13.15",
    )
    python_change = _artifact_fingerprint(
        "source-artifact",
        "source-lock",
        "parser-code",
        {"pypdf": "6.19.0", "fonttools": "4.58.0", "ftfy": "6.3.1"},
        "3.13.16",
    )
    fonttools_change = _artifact_fingerprint(
        "source-artifact",
        "source-lock",
        "parser-code",
        {"pypdf": "6.19.0", "fonttools": "4.59.0", "ftfy": "6.3.1"},
        "3.13.15",
    )
    assert len(base) == 64
    assert base != package_change
    assert base != python_change
    assert base != fonttools_change


def test_artifact_fingerprint_changes_with_parent_source_artifact():
    first = _artifact_fingerprint(
        "source-a",
        "source-lock",
        "parser-code",
        {"pypdf": "6.14.2"},
        "3.13.15",
    )
    second = _artifact_fingerprint(
        "source-b",
        "source-lock",
        "parser-code",
        {"pypdf": "6.14.2"},
        "3.13.15",
    )
    assert first != second
