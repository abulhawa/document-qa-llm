"""Pure validation tests for the benchmark cache seeder."""

import struct

import pytest

from scripts.seed_benchmark_cache import checked_path, document_digest, npy_shape, signature


def test_stage_signature_tracks_configuration_but_not_corpus_fingerprint():
    manifest = {
        "parser_fingerprint": "code-v1",
        "parser_config": {"limit": 50000},
        "package_versions": {"pypdf": "6.0"},
        "python_version": "3.13.0",
        "parent_source_artifact_fingerprint": "corpus-a",
    }
    original = signature(manifest, "parsed")
    assert signature({**manifest, "parent_source_artifact_fingerprint": "corpus-b"}, "parsed") == original
    assert signature({**manifest, "parser_config": {"limit": 60000}}, "parsed") != original


def test_parsed_content_hash_excludes_benchmark_specific_source_uri():
    doc = {"filetype": "pdf"}
    page = {"page_index": 0, "text": "same text", "metadata": {"source": "benchmark://v1/doc", "page": 1}}
    other = {**page, "metadata": {**page["metadata"], "source": "benchmark://v2/doc"}}
    assert document_digest(doc, [page]) == document_digest(doc, [other])
    assert document_digest(doc, [page]) != document_digest(doc, [{**page, "text": "changed"}])


def test_npy_header_checks_shape_and_payload_size():
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2), }"
    header += b" " * ((16 - (10 + len(header) + 1) % 16) % 16) + b"\n"
    data = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<ff", 1.0, 2.0)
    assert npy_shape(data) == (1, 2, 10 + len(header))
    with pytest.raises(ValueError, match="byte count"):
        npy_shape(data[:-1])


@pytest.mark.parametrize("path", ["/root/file", "../file", "root/../file"])
def test_bucket_path_validation_rejects_traversal(path):
    with pytest.raises(ValueError, match="unsafe bucket path"):
        checked_path(path)

