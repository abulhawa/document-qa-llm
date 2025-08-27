import os
from langchain_core.documents import Document

from core.ingestion import ingest_one


def test_ingest_one_returns_normalized_path(tmp_path, monkeypatch):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello")

    # Stub out external dependencies used during ingestion
    monkeypatch.setattr(
        "core.ingestion.load_documents",
        lambda p: [Document(page_content="hello", metadata={})],
    )
    monkeypatch.setattr(
        "core.ingestion.preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: docs_like,
    )
    monkeypatch.setattr(
        "core.ingestion.split_documents", lambda docs: [{"text": "hello"}]
    )
    monkeypatch.setattr("core.ingestion.index_documents", lambda chunks: None)
    monkeypatch.setattr("utils.qdrant_utils.index_chunks", lambda chunks: True)
    monkeypatch.setattr(
        "core.ingestion.is_file_up_to_date", lambda checksum, path: False
    )
    monkeypatch.setattr(
        "core.ingestion.is_duplicate_checksum", lambda checksum, path: False
    )
    monkeypatch.setattr(
        "core.ingestion.index_fulltext_document", lambda doc: None
    )

    result = ingest_one(str(sample))

    assert result["path"] == os.path.normpath(str(sample)).replace("\\", "/")
    assert result["success"]

