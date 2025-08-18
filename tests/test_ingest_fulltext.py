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


def test_fulltext_index_called(tmp_path, monkeypatch):
    f = tmp_path / "doc.txt"
    f.write_text("hello")

    monkeypatch.setattr("core.ingestion.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("core.ingestion.IngestLogEmitter", DummyLog)
    monkeypatch.setattr("core.ingestion.get_file_size", lambda p: 10)
    monkeypatch.setattr(
        "core.ingestion.get_file_timestamps",
        lambda p: {"created": "2023-01-01", "modified": "2023-01-01"},
    )

    monkeypatch.setattr(
        "core.ingestion.load_documents", lambda p: [Document(page_content="full", metadata={})]
    )
    monkeypatch.setattr(
        "core.ingestion.preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: [
            Document(page_content="full", metadata={})
        ],
    )
    monkeypatch.setattr("core.ingestion.split_documents", lambda docs: [{"text": "chunk"}])
    monkeypatch.setattr("core.ingestion.index_documents", lambda chunks: None)
    monkeypatch.setattr(
        "core.ingestion.set_has_embedding_true_by_ids", lambda ids: (0, 0)
    )

    called = {}

    def fake_index_fulltext(doc):
        called["doc"] = doc

    monkeypatch.setattr("core.ingestion.index_fulltext_document", fake_index_fulltext)

    ingest_one(str(f))

    assert "doc" in called
    assert called["doc"]["text_full"] == "full"
    assert called["doc"]["path"] == str(f)
    assert called["doc"]["checksum"] == "abc"

