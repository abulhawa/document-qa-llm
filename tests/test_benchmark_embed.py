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


def test_local_model_snapshot_requires_expected_files(tmp_path):
    from scripts import embed_benchmark_chunks as module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("modules.json", "config.json", "model.safetensors"):
        (snapshot / name).write_text("x", encoding="utf-8")

    assert module._validate_local_model_snapshot(snapshot) == snapshot


def test_local_model_snapshot_rejects_incomplete_cache(tmp_path):
    from scripts import embed_benchmark_chunks as module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("x", encoding="utf-8")

    try:
        module._validate_local_model_snapshot(snapshot)
    except FileNotFoundError as exc:
        assert "incomplete" in str(exc)
        assert "modules.json" in str(exc)
        assert "model.safetensors" in str(exc)
    else:
        raise AssertionError("incomplete cached model should fail")


def test_local_model_resolution_uses_slim_snapshot_contract(tmp_path, monkeypatch):
    import sys
    import types

    from scripts import embed_benchmark_chunks as module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("modules.json", "config.json", "model.safetensors"):
        (snapshot / name).write_text("x", encoding="utf-8")

    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(snapshot)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    assert module._resolve_local_model_snapshot("org/model", "deadbeef") == snapshot
    assert captured["repo_id"] == "org/model"
    assert captured["revision"] == "deadbeef"
    assert captured["local_files_only"] is True
    assert captured["ignore_patterns"] == module.MODEL_IGNORE_PATTERNS
