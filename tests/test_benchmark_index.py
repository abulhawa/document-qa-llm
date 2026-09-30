from scripts.build_benchmark_indexes import (
    _backend_chunk_id,
    _chunk_source,
    _qdrant_payload,
    checkpoint_signature,
    combined_index_fingerprint,
    opensearch_engine_fingerprint,
    qdrant_engine_fingerprint,
)


def test_opensearch_fingerprint_tracks_chunks_not_embeddings():
    settings = {"settings": {"index": {"number_of_replicas": 0}}, "mappings": {}}
    base = opensearch_engine_fingerprint("chunks-a", "3.8.0", settings=settings)
    assert len(base) == 64
    assert base != opensearch_engine_fingerprint("chunks-b", "3.8.0", settings=settings)
    assert base != opensearch_engine_fingerprint("chunks-a", "3.9.0", settings=settings)


def test_qdrant_fingerprint_tracks_embeddings_and_dimension():
    base = qdrant_engine_fingerprint("chunks-a", "embed-a", "1.19.1", 768)
    assert len(base) == 64
    assert base != qdrant_engine_fingerprint("chunks-a", "embed-b", "1.19.1", 768)
    assert base != qdrant_engine_fingerprint("chunks-a", "embed-a", "1.19.1", 1024)


def test_checkpoint_signature_is_separate_from_engine_identity():
    engine = qdrant_engine_fingerprint("chunks", "embed", "1.19.1", 768)
    a = checkpoint_signature(engine, "builder-a")
    b = checkpoint_signature(engine, "builder-b")
    assert a != b
    assert engine == qdrant_engine_fingerprint("chunks", "embed", "1.19.1", 768)


def test_combined_fingerprint_tracks_each_engine():
    base = combined_index_fingerprint(
        benchmark_id="composite-v1",
        chunks_fingerprint="chunks",
        embeddings_fingerprint="embed",
        opensearch_fingerprint="os-a",
        qdrant_fingerprint="qd-a",
    )
    changed = combined_index_fingerprint(
        benchmark_id="composite-v1",
        chunks_fingerprint="chunks",
        embeddings_fingerprint="embed",
        opensearch_fingerprint="os-a",
        qdrant_fingerprint="qd-b",
    )
    assert len(base) == 64
    assert base != changed


def test_index_payload_contract_uses_stable_chunk_identity():
    row = {
        "id": "chunk-1",
        "text": "hello",
        "path": "benchmark://composite-v1/nfcorpus/doc-1",
        "chunk_index": 3,
        "chunk_char_len": 5,
        "source_sha256": "abc123",
        "filetype": "text",
        "page": 2,
        "location_percent": 25.0,
    }
    source = _chunk_source(row)
    payload = _qdrant_payload(row)
    assert source["checksum"] == "abc123"
    assert source["text"] == "hello"
    assert payload == {
        "id": _backend_chunk_id(row),
        "checksum": "abc123",
        "path": "benchmark://composite-v1/nfcorpus/doc-1",
    }


def test_backend_chunk_id_preserves_duplicate_content_provenance():
    base = {
        "id": "same-content-chunk",
        "chunk_index": 0,
    }
    a = {
        **base,
        "path": "benchmark://composite-v1/open_ragbench/doc-a",
    }
    b = {
        **base,
        "path": "benchmark://composite-v1/officeqa/doc-b",
    }

    assert _backend_chunk_id(a) != _backend_chunk_id(b)
    assert _backend_chunk_id(a) == _backend_chunk_id(a)


def test_incremental_opensearch_restore_makes_snapshot_directories_writable(tmp_path):
    from scripts.incremental_benchmark_indexes import _make_repo_directories_writable

    nested = tmp_path / "indices" / "abc" / "0"
    nested.mkdir(parents=True)
    for path in (tmp_path / "indices", tmp_path / "indices" / "abc", nested):
        path.chmod(0o555)

    _make_repo_directories_writable(tmp_path)

    for path in (tmp_path, tmp_path / "indices", tmp_path / "indices" / "abc", nested):
        assert path.stat().st_mode & 0o222 == 0o222
