import os
import types

import pytest

from core.ingestion import ingest_one


class DummyLog:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set(self, **kwargs):
        pass

    def done(self, **kwargs):
        pass

    def fail(self, **kwargs):
        pass


# Test 21: idempotent skip when file unchanged

def test_ingest_one_idempotent_skip(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("core.ingestion.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda c, p: True)
    monkeypatch.setattr("core.ingestion.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.IngestLogEmitter", DummyLog)

    # ensure load_documents not called
    def boom(path):
        raise AssertionError("should not load")

    monkeypatch.setattr("core.ingestion.load_documents", boom)

    result = ingest_one(str(f))
    assert result["status"] == "Already indexed"
    assert result["success"] is False


# Test 23: Embedder failure marks log and skips flag flip

def test_ingest_one_embedder_failure(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("core.ingestion.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.IngestLogEmitter", DummyLog)

    monkeypatch.setattr("core.ingestion.load_documents", lambda p: ["doc"])  # one doc
    monkeypatch.setattr(
        "core.ingestion.preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: docs_like,
    )
    monkeypatch.setattr("core.ingestion.split_documents", lambda docs: [{"text": "hello"}])
    monkeypatch.setattr("core.ingestion.index_documents", lambda chunks: None)

    monkeypatch.setattr("utils.qdrant_utils.index_chunks", lambda chunks: False)

    called = {"flip": False}

    def fake_flip(ids):
        called["flip"] = True
        return (0, 0)

    monkeypatch.setattr("core.ingestion.set_has_embedding_true_by_ids", fake_flip)

    result = ingest_one(str(f))
    assert result["success"] is False
    assert result["status"] == "Local indexing failed"
    assert called["flip"] is False


# Test 24: batching behavior for large files

def test_ingest_one_batching(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("core.ingestion.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.IngestLogEmitter", DummyLog)

    monkeypatch.setattr("core.ingestion.load_documents", lambda p: ["doc"])  # one doc
    monkeypatch.setattr(
        "core.ingestion.preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: docs_like,
    )
    # produce 5 chunks
    monkeypatch.setattr(
        "core.ingestion.split_documents", lambda docs: [{"text": str(i)} for i in range(5)]
    )
    monkeypatch.setattr("core.ingestion.index_documents", lambda chunks: None)

    # Force large path
    monkeypatch.setattr("core.ingestion.CHUNK_EMBEDDING_THRESHOLD", 2)
    monkeypatch.setattr("core.ingestion.EMBEDDING_BATCH_SIZE", 2)

    calls = []

    class DummyApp:
        def send_task(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr("core.ingestion.celery_app", DummyApp())

    result = ingest_one(str(f))
    assert len(calls) == 3


# Test 25: many files trigger background indexing even if chunks are small
def test_ingest_one_background_many_files(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("core.ingestion.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.IngestLogEmitter", DummyLog)

    monkeypatch.setattr("core.ingestion.load_documents", lambda p: ["doc"])  # one doc
    monkeypatch.setattr(
        "core.ingestion.preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: docs_like,
    )
    monkeypatch.setattr("core.ingestion.split_documents", lambda docs: [{"text": "hello"}])

    # Ensure thresholds trigger on total_files
    monkeypatch.setattr("core.ingestion.CHUNK_EMBEDDING_THRESHOLD", 100)
    monkeypatch.setattr("core.ingestion.MAX_FILES_FOR_FULL_EMBEDDING", 1)

    # Prevent local indexing
    def boom(chunks):
        raise AssertionError("should not index locally")

    monkeypatch.setattr("core.ingestion.index_documents", boom)

    calls = []

    class DummyApp:
        def send_task(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr("core.ingestion.celery_app", DummyApp())

    result = ingest_one(str(f), total_files=2)
    assert len(calls) == 1

