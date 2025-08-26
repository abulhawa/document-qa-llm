import os
import types

import pytest
from langchain_core.documents import Document

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
    assert result["success"] is True


# Test 23: Embedder failure marks log and skips flag flip

def test_ingest_one_embedder_failure(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("core.ingestion.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.IngestLogEmitter", DummyLog)

    monkeypatch.setattr(
        "core.ingestion.load_documents",
        lambda p: [Document(page_content="doc", metadata={})],
    )  # one doc
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


def test_ingest_one_handles_multiple_chunks(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("core.ingestion.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.IngestLogEmitter", DummyLog)

    monkeypatch.setattr(
        "core.ingestion.load_documents",
        lambda p: [Document(page_content="doc", metadata={})],
    )
    monkeypatch.setattr(
        "core.ingestion.preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: docs_like,
    )
    monkeypatch.setattr(
        "core.ingestion.split_documents", lambda docs: [{"text": str(i)} for i in range(5)]
    )

    captured = {}

    def fake_index_documents(chunks):
        captured["count"] = len(chunks)

    monkeypatch.setattr("core.ingestion.index_documents", fake_index_documents)
    monkeypatch.setattr("utils.qdrant_utils.index_chunks", lambda chunks: True)
    monkeypatch.setattr("core.ingestion.index_fulltext_document", lambda doc: None)
    monkeypatch.setattr(
        "core.ingestion.set_has_embedding_true_by_ids", lambda ids: (0, 0)
    )

    result = ingest_one(str(f))
    assert result["success"] is True
    assert captured["count"] == 5


def test_ingest_one_background_many_files(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("core.ingestion.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.IngestLogEmitter", DummyLog)

    monkeypatch.setattr(
        "core.ingestion.load_documents",
        lambda p: [Document(page_content="doc", metadata={})],
    )
    monkeypatch.setattr(
        "core.ingestion.preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: docs_like,
    )
    monkeypatch.setattr("core.ingestion.split_documents", lambda docs: [{"text": "hello"}])
    monkeypatch.setattr("core.ingestion.index_documents", lambda chunks: None)
    monkeypatch.setattr("utils.qdrant_utils.index_chunks", lambda chunks: True)
    monkeypatch.setattr("core.ingestion.index_fulltext_document", lambda doc: None)
    monkeypatch.setattr(
        "core.ingestion.set_has_embedding_true_by_ids", lambda ids: (0, 0)
    )

    result = ingest_one(str(f), total_files=2)
    assert result["success"] is True

