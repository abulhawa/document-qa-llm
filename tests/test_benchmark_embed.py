from scripts.embed_benchmark_chunks import _artifact_fingerprint
from scripts.finalize_benchmark_embeddings import finalize


def test_embedding_fingerprint_tracks_parent_revision_and_input_contract():
    packages = {"sentence-transformers": "5.4.0", "torch": "2.11.0"}
    base = _artifact_fingerprint(
        "chunks-a", "code", packages, "3.13.15",
        "intfloat/multilingual-e5-base", "revision-a", "e5",
        True, "float32", "cpu",
    )
    parent_change = _artifact_fingerprint(
        "chunks-b", "code", packages, "3.13.15",
        "intfloat/multilingual-e5-base", "revision-a", "e5",
        True, "float32", "cpu",
    )
    revision_change = _artifact_fingerprint(
        "chunks-a", "code", packages, "3.13.15",
        "intfloat/multilingual-e5-base", "revision-b", "e5",
        True, "float32", "cpu",
    )
    input_change = _artifact_fingerprint(
        "chunks-a", "code", packages, "3.13.15",
        "intfloat/multilingual-e5-base", "revision-a", "raw",
        True, "float32", "cpu",
    )
    assert len(base) == 64
    assert base != parent_change
    assert base != revision_change
    assert base != input_change


def test_finalize_rejects_missing_shards(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    try:
        finalize(
            markers,
            tmp_path / "manifest.json",
            benchmark_id="composite-v1",
            parent_chunks_fingerprint="chunks",
            artifact_uri="hf://bucket/path",
            expected_shards=2,
        )
    except ValueError as exc:
        assert "expected 2 shard markers" in str(exc)
    else:
        raise AssertionError("missing shards should fail")
