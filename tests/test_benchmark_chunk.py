from scripts.chunk_benchmark_parsed import (
    _artifact_fingerprint,
    _chunk_document,
    _chunker_fingerprint,
)


def test_chunker_fingerprint_is_stable():
    first = _chunker_fingerprint()
    second = _chunker_fingerprint()
    assert first == second
    assert len(first) == 64


def test_chunk_artifact_fingerprint_tracks_parent_config_and_runtime():
    base = _artifact_fingerprint(
        "parsed-a",
        "chunker-code",
        {"langchain-core": "1.3.3", "langchain-text-splitters": "1.1.0"},
        "3.13.15",
        800,
        100,
    )
    parent_change = _artifact_fingerprint(
        "parsed-b",
        "chunker-code",
        {"langchain-core": "1.3.3", "langchain-text-splitters": "1.1.0"},
        "3.13.15",
        800,
        100,
    )
    config_change = _artifact_fingerprint(
        "parsed-a",
        "chunker-code",
        {"langchain-core": "1.3.3", "langchain-text-splitters": "1.1.0"},
        "3.13.15",
        600,
        80,
    )
    package_change = _artifact_fingerprint(
        "parsed-a",
        "chunker-code",
        {"langchain-core": "1.3.3", "langchain-text-splitters": "1.1.1"},
        "3.13.15",
        800,
        100,
    )
    assert len(base) == 64
    assert base != parent_change
    assert base != config_change
    assert base != package_change


def test_chunk_document_uses_production_splitter_and_file_level_indices():
    document = {
        "source_document_id": "doc-1",
        "source_sha256": "abc123",
        "filetype": "txt",
    }
    pages = [
        {
            "source_document_id": "doc-1",
            "page_index": 0,
            "text": "alpha " * 180,
            "metadata": {},
        },
        {
            "source_document_id": "doc-1",
            "page_index": 1,
            "text": "beta " * 180,
            "metadata": {},
        },
    ]
    chunks = _chunk_document(
        benchmark_id="composite-v1",
        track="example",
        document=document,
        pages=pages,
        chunk_size=400,
        chunk_overlap=50,
    )
    assert len(chunks) > 2
    assert [row["chunk_index"] for row in chunks] == list(range(len(chunks)))
    assert all(row["text"] for row in chunks)
    assert all(row["path"] == "benchmark://composite-v1/example/doc-1" for row in chunks)
    assert chunks[0]["location_percent"] == 0
    assert chunks[-1]["location_percent"] == 100
