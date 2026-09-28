from scripts.parse_benchmark_source import _normalise_metadata, _parser_fingerprint


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
