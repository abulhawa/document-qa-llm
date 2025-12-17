import os

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


def test_fulltext_index_called(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("utils.file_utils.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("ingestion.storage.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr(
        "ingestion.storage.is_duplicate_checksum", lambda c, p: False
    )
    monkeypatch.setattr(
        "ingestion.orchestrator.IngestLogEmitter", DummyLog
    )
    monkeypatch.setattr("utils.file_utils.get_file_size", lambda p: 10)
    monkeypatch.setattr(
        "utils.file_utils.get_file_timestamps",
        lambda p: {"created": "2023-01-01", "modified": "2023-01-01"},
    )

    monkeypatch.setattr(
        "ingestion.io_loader.load_file_documents",
        lambda p: [Document(page_content="full", metadata={})],
    )
    monkeypatch.setattr(
        "ingestion.preprocess.preprocess_documents",
        lambda docs_like, normalized_path, ext: [
            Document(page_content="full", metadata={})
        ],
    )
    monkeypatch.setattr(
        "ingestion.preprocess.chunk_documents", lambda docs: [{"text": "chunk"}]
    )
    monkeypatch.setattr(
        "ingestion.storage.index_chunk_batch", lambda chunks: (len(chunks), [])
    )
    monkeypatch.setattr(
        "utils.qdrant_utils.index_chunks_in_batches",
        lambda chunks, os_index_batch=None: True,
    )

    called = {}

    def fake_index_fulltext(doc):
        called["doc"] = doc

    monkeypatch.setattr("ingestion.storage.index_fulltext", fake_index_fulltext)

    result = ingest_one(str(f))

    assert result["success"] is True
    assert "doc" in called
    assert called["doc"]["text_full"] == "full"
    assert os.path.normpath(called["doc"]["path"]) == os.path.normpath(str(f))
    assert called["doc"]["checksum"] == "abc"

