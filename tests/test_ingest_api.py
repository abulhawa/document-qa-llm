import os
from langchain_core.documents import Document

from ingestion.orchestrator import ingest_one


def test_ingest_one_returns_normalized_path(tmp_path, monkeypatch):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello")

    # Stub out external dependencies used during ingestion
    monkeypatch.setattr(
        "ingestion.io_loader.load_file_documents",
        lambda p: [Document(page_content="hello", metadata={})],
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
        "ingestion.storage.embed_and_store",
        lambda chunks, os_index_batch=None: True,
    )
    monkeypatch.setattr(
        "ingestion.storage.is_file_up_to_date", lambda checksum, path: False
    )
    monkeypatch.setattr(
        "ingestion.storage.is_duplicate_checksum", lambda checksum, path: False
    )
    monkeypatch.setattr("ingestion.storage.index_fulltext", lambda doc: None)

    result = ingest_one(str(sample))

    assert result["path"] == os.path.normpath(str(sample)).replace("\\", "/")
    assert result["success"]

