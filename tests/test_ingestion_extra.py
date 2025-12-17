import os
import types

import pytest
from langchain_core.documents import Document

from ingestion.orchestrator import ingest_one


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

    monkeypatch.setattr("utils.file_utils.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("ingestion.storage.is_file_up_to_date", lambda c, p: True)
    monkeypatch.setattr("ingestion.storage.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("ingestion.orchestrator.IngestLogEmitter", DummyLog)

    def boom(path):
        raise AssertionError("should not load")

    monkeypatch.setattr("ingestion.io_loader.load_file_documents", boom)

    result = ingest_one(str(f))
    assert result["status"] == "Already indexed"
    assert result["success"] is True


# Test 23: Embedder failure marks log


def test_ingest_one_embedder_failure(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("utils.file_utils.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("ingestion.storage.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("ingestion.storage.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("ingestion.orchestrator.IngestLogEmitter", DummyLog)

    monkeypatch.setattr(
        "ingestion.io_loader.load_file_documents",
        lambda p: [Document(page_content="doc", metadata={})],
    )  # one doc
    monkeypatch.setattr(
        "ingestion.preprocess.preprocess_documents",
        lambda docs_like, normalized_path, ext: docs_like,
    )
    monkeypatch.setattr(
        "ingestion.preprocess.chunk_documents", lambda docs: [{"text": "hello"}]
    )
    monkeypatch.setattr(
        "ingestion.storage.index_chunk_batch", lambda chunks: (len(chunks), [])
    )

    monkeypatch.setattr(
        "utils.qdrant_utils.index_chunks_in_batches",
        lambda chunks, os_index_batch=None: False,
    )

    with pytest.raises(RuntimeError):
        ingest_one(str(f))


def test_ingest_one_handles_multiple_chunks(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("utils.file_utils.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("ingestion.storage.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("ingestion.storage.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("ingestion.orchestrator.IngestLogEmitter", DummyLog)

    monkeypatch.setattr(
        "ingestion.io_loader.load_file_documents",
        lambda p: [Document(page_content="doc", metadata={})],
    )
    monkeypatch.setattr(
        "ingestion.preprocess.preprocess_documents",
        lambda docs_like, normalized_path, ext: docs_like,
    )
    monkeypatch.setattr(
        "ingestion.preprocess.chunk_documents",
        lambda docs: [{"text": str(i)} for i in range(5)],
    )

    captured = {}

    def fake_index_documents(chunks):
        captured["count"] = len(chunks)
        return len(chunks), []

    monkeypatch.setattr("ingestion.storage.index_chunk_batch", fake_index_documents)

    def fake_index_chunks(chunks, os_index_batch=None):
        if os_index_batch:
            os_index_batch(chunks)
        return True

    monkeypatch.setattr(
        "utils.qdrant_utils.index_chunks_in_batches", fake_index_chunks
    )
    monkeypatch.setattr("ingestion.storage.index_fulltext", lambda doc: None)

    result = ingest_one(str(f))
    assert result["success"] is True
    assert captured["count"] == 5


def test_ingest_one_background_many_files(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("utils.file_utils.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("ingestion.storage.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("ingestion.storage.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("ingestion.orchestrator.IngestLogEmitter", DummyLog)

    monkeypatch.setattr(
        "ingestion.io_loader.load_file_documents",
        lambda p: [Document(page_content="doc", metadata={})],
    )
    monkeypatch.setattr(
        "ingestion.preprocess.preprocess_documents",
        lambda docs_like, normalized_path, ext: docs_like,
    )
    monkeypatch.setattr(
        "ingestion.preprocess.chunk_documents", lambda docs: [{"text": "hello"}]
    )
    monkeypatch.setattr(
        "ingestion.storage.index_chunk_batch", lambda chunks: (len(chunks), [])
    )
    monkeypatch.setattr(
        "utils.qdrant_utils.index_chunks_in_batches",
        lambda chunks, os_index_batch=None: True,
    )
    monkeypatch.setattr("ingestion.storage.index_fulltext", lambda doc: None)

    result = ingest_one(str(f), total_files=2)
    assert result["success"] is True
