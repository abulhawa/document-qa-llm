"""Behavior checks for cross-corpus reuse without external services."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import chunk_benchmark_parsed as chunker
from scripts import embed_benchmark_chunks as embedder
from scripts import incremental_benchmark_stages as incremental
from scripts import parse_benchmark_source as parser


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _manifest(root: Path, stage: str, **extra: object) -> None:
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            data = path.read_bytes()
            files.append({"path": path.relative_to(root).as_posix(),
                          "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    _json(root / "manifest.json", {
        "artifact_type": stage, "artifact_fingerprint": f"{stage}-v1",
        "benchmark_id": "composite-v1", "persistence": {"persisted": True},
        "files": files, **extra,
    })


def _empty_tracks(root: Path, names: tuple[str, ...]) -> None:
    for track in parser.TRACKS:
        for name in names:
            _jsonl(root / track / name, [])


def test_parse_reuses_unchanged_document_without_calling_loader(tmp_path, monkeypatch) -> None:
    base_source = tmp_path / "base-source"
    base_parsed = tmp_path / "base-parsed"
    target_source = tmp_path / "target-source"
    _empty_tracks(base_source, ("documents.jsonl",))
    _empty_tracks(base_parsed, ("documents.jsonl", "pages.jsonl"))
    _empty_tracks(target_source, ("documents.jsonl",))
    source_row = {"source_document_id": "doc-a", "path": "documents/doc-a.pdf",
                  "sha256": "source-hash", "bytes": 123}
    parsed_row = {"source_benchmark": "open_ragbench", "source_document_id": "doc-a",
                  "source_path": source_row["path"], "filetype": "pdf",
                  "source_sha256": "source-hash", "source_bytes": 123,
                  "page_count": 1, "empty_pages": 0, "character_count": 5, "status": "parsed"}
    page = {"source_benchmark": "open_ragbench", "source_document_id": "doc-a",
            "page_index": 0, "text": "hello",
            "metadata": {"source": "benchmark://composite-v1/open_ragbench/doc-a"}}
    _jsonl(base_source / "open_ragbench" / "documents.jsonl", [source_row])
    _jsonl(base_parsed / "open_ragbench" / "documents.jsonl", [parsed_row])
    _jsonl(base_parsed / "open_ragbench" / "pages.jsonl", [page])
    _jsonl(target_source / "open_ragbench" / "documents.jsonl", [source_row])
    _json(target_source / "source.lock.json", {})
    # The source manifest maps the raw PDF hash without requiring a second download.
    _manifest(base_source, "source")
    manifest = json.loads((base_source / "manifest.json").read_text())
    manifest["files"].append({"path": "open_ragbench/documents/doc-a.pdf",
                              "sha256": "source-hash", "bytes": 123})
    _json(base_source / "manifest.json", manifest)
    _manifest(base_parsed, "parsed",
              parent_source_artifact_fingerprint="source-v1",
              parser_fingerprint=parser._parser_fingerprint(),
              parser_config={"pypdf_xform_maximum_invocations_per_extraction": 50000},
              package_versions=parser._package_versions(),
              python_version=incremental.platform.python_version())
    monkeypatch.setattr(parser, "_parse_one", lambda task: (_ for _ in ()).throw(AssertionError("loader called")))
    output = tmp_path / "parsed-out"
    incremental.parse(argparse.Namespace(
        base_source_root=base_source, base_root=base_parsed, source_root=target_source,
        output_root=output, benchmark_id="composite-v2", workers=1, max_failures=0,
        repo_revision="test", parent_fingerprint="source-v2", pypdf_xform_limit=50000,
    ))
    result = json.loads((output / "composite-v2" / "manifest.json").read_text())
    assert result["reuse"]["documents_reused"] == 1
    assert result["reuse"]["documents_computed"] == 0


def test_chunk_reuses_same_parsed_content_with_predecessor_path(tmp_path, monkeypatch) -> None:
    base_parsed = tmp_path / "base-parsed"
    base_chunks = tmp_path / "base-chunks"
    target_parsed = tmp_path / "target-parsed"
    _empty_tracks(base_parsed, ("documents.jsonl", "pages.jsonl"))
    _empty_tracks(base_chunks, ("documents.jsonl", "chunks.jsonl"))
    _empty_tracks(target_parsed, ("documents.jsonl", "pages.jsonl"))
    doc = {"source_benchmark": "open_ragbench", "source_document_id": "doc-a",
           "source_sha256": "source-hash", "filetype": "pdf", "page_count": 1,
           "empty_pages": 0, "status": "parsed"}
    page = {"source_benchmark": "open_ragbench", "source_document_id": "doc-a",
            "page_index": 0, "text": "hello", "metadata": {"source": "benchmark://composite-v1/open_ragbench/doc-a"}}
    row = {"id": "chunk-a", "source_benchmark": "open_ragbench",
           "source_document_id": "doc-a", "source_sha256": "source-hash",
           "text": "hello", "chunk_char_len": 5, "chunk_index": 0, "page": 0,
           "path": "benchmark://composite-v1/open_ragbench/doc-a",
           "location_percent": 0, "filetype": "pdf"}
    for root in (base_parsed, target_parsed):
        _jsonl(root / "open_ragbench" / "documents.jsonl", [doc])
        _jsonl(root / "open_ragbench" / "pages.jsonl", [
            {**page, "metadata": {"source": f"benchmark://{'composite-v1' if root == base_parsed else 'composite-v2'}/open_ragbench/doc-a"}}
        ])
    _jsonl(base_chunks / "open_ragbench" / "documents.jsonl", [
        {"source_benchmark": "open_ragbench", "source_document_id": "doc-a",
         "source_sha256": "source-hash", "page_count": 1, "empty_pages": 0,
         "chunk_count": 1, "status": "chunked"}
    ])
    _jsonl(base_chunks / "open_ragbench" / "chunks.jsonl", [row])
    _manifest(base_parsed, "parsed")
    _manifest(base_chunks, "chunks",
              parent_parsed_artifact_fingerprint="parsed-v1",
              chunker_fingerprint=chunker._chunker_fingerprint(),
              chunker_config={"chunk_size": 800, "chunk_overlap": 100},
              package_versions=chunker._package_versions(),
              python_version=incremental.platform.python_version())
    _json(target_parsed / "manifest.json", {
        "artifact_type": "parsed", "artifact_fingerprint": "parsed-v2",
        "benchmark_id": "composite-v2", "persistence": {"persisted": True},
    })
    monkeypatch.setattr(chunker, "_chunk_document", lambda **kwargs: (_ for _ in ()).throw(AssertionError("splitter called")))
    output = tmp_path / "chunks-out"
    incremental.chunk(argparse.Namespace(
        base_parsed_root=base_parsed, base_root=base_chunks, parsed_root=target_parsed,
        output_root=output, benchmark_id="composite-v2", parent_fingerprint="parsed-v2",
        chunk_size=800, chunk_overlap=100, max_failures=0, repo_revision="test",
    ))
    result = json.loads((output / "composite-v2" / "manifest.json").read_text())
    assert result["reuse"]["documents_reused"] == 1
    assert result["reuse"]["chunks_reused"] == 1
    assert json.loads((output / "composite-v2" / "open_ragbench" / "chunks.jsonl").read_text())[
        "path"
    ] == row["path"]


def test_embedding_reuses_exact_old_vector_and_encodes_only_new_text(tmp_path, monkeypatch) -> None:
    import numpy as np

    base_chunks = tmp_path / "base-chunks"
    base_embeddings = tmp_path / "base-embeddings"
    target_chunks = tmp_path / "target-chunks"
    for root in (base_chunks, target_chunks):
        _empty_tracks(root, ("chunks.jsonl",))
    old = {"id": "old-chunk", "source_benchmark": "open_ragbench",
           "source_document_id": "doc-a", "chunk_index": 0, "text": "old"}
    new = {"id": "new-chunk", "source_benchmark": "open_ragbench",
           "source_document_id": "doc-b", "chunk_index": 0, "text": "new"}
    _jsonl(base_chunks / "open_ragbench" / "chunks.jsonl", [old])
    _jsonl(target_chunks / "open_ragbench" / "chunks.jsonl", [old, new])
    _manifest(base_chunks, "chunks")
    shard = base_embeddings / "shards" / "000"
    shard.mkdir(parents=True)
    np.save(shard / "embeddings.npy", np.asarray([[1.0, 2.0]], dtype=np.float32), allow_pickle=False)
    _jsonl(shard / "records.jsonl", [{"row_index": 0, "global_index": 0,
                                     "id": "old-chunk", "source_benchmark": "open_ragbench",
                                     "source_document_id": "doc-a", "chunk_index": 0}])
    model = {"name": "test-model", "revision": "revision", "input_format": "e5",
             "document_input_type": "passage", "query_input_type": "query",
             "normalize_embeddings": True, "execution_device": "cpu", "output_dtype": "float32"}
    _manifest(base_embeddings, "embeddings", parent_chunks_artifact_fingerprint="chunks-v1",
              embedder_fingerprint=embedder._embedder_fingerprint(), model=model,
              package_versions=embedder._package_versions(),
              python_version=incremental.platform.python_version(),
              shard_count=1, vector_dimension=2)
    calls: list[list[str]] = []

    def encode(texts, **kwargs):
        calls.append(texts)
        return np.asarray([[3.0, 4.0]], dtype=np.float32)

    def build(chunks_root, output_root, checkpoint_root, **kwargs):
        vectors = kwargs["vector_provider"]([old, new], None, batch_size=16, input_format="e5")
        assert vectors.tolist() == [[1.0, 2.0], [3.0, 4.0]]
        output = output_root / "composite-v2"
        _json(output / "shards" / "000" / "manifest.json", {"artifact_type": "embedding-shard"})
        return output

    monkeypatch.setattr(embedder, "_encode", encode)
    monkeypatch.setattr(incremental.checkpoint, "build_checkpointed_shard", build)
    output = tmp_path / "embedding-out"
    incremental.embed(argparse.Namespace(
        base_chunks_root=base_chunks, base_root=base_embeddings,
        chunks_root=target_chunks, output_root=output, checkpoint_root=tmp_path / "checkpoints",
        benchmark_id="composite-v2", parent_fingerprint="chunks-v2",
        artifact_fingerprint="embedding-v2", model_name="test-model",
        model_revision="revision", input_format="e5", execution_device="cpu",
        batch_size=16, shard_index=0, shard_count=1, checkpoint_size=2048,
        remote_root="hf://test", fingerprint_only=False, repo_revision="test",
    ))
    reuse = json.loads((output / "composite-v2" / "shards" / "000" / "manifest.json").read_text())["reuse"]
    assert reuse["vectors_reused"] == 1
    assert reuse["vectors_computed"] == 1
    assert calls == [["new"]]

