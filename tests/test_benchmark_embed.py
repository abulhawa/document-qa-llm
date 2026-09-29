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


def test_checkpoint_ranges_cover_rows_without_overlap():
    from scripts.checkpoint_benchmark_embeddings import _part_ranges

    assert _part_ranges(0, 2048) == []
    assert _part_ranges(5000, 2048) == [
        (0, 2048),
        (2048, 4096),
        (4096, 5000),
    ]


def test_checkpoint_signature_tracks_execution_contract():
    from scripts.checkpoint_benchmark_embeddings import _checkpoint_signature

    kwargs = {
        "artifact_fingerprint": "artifact-a",
        "shard_count": 8,
        "checkpoint_size": 2048,
        "batch_size": 16,
        "helper_fingerprint": "helper-a",
    }
    base = _checkpoint_signature(**kwargs)
    assert len(base) == 64
    assert base != _checkpoint_signature(
        **{**kwargs, "artifact_fingerprint": "artifact-b"}
    )
    assert base != _checkpoint_signature(
        **{**kwargs, "checkpoint_size": 4096}
    )
    assert base != _checkpoint_signature(
        **{**kwargs, "batch_size": 32}
    )
    assert base != _checkpoint_signature(
        **{**kwargs, "helper_fingerprint": "helper-b"}
    )


def test_checkpoint_hf_cli_can_reach_bucket_while_model_loading_is_offline(monkeypatch):
    import subprocess

    from scripts import checkpoint_benchmark_embeddings as module

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module._hf(["cp", "source", "destination"])

    assert captured["command"] == ["hf", "buckets", "cp", "source", "destination"]
    assert "HF_HUB_OFFLINE" not in captured["env"]
    assert captured["env"]["TRANSFORMERS_OFFLINE"] == "1"
