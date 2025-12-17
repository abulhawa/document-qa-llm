import os
from langchain_core.documents import Document

from ingestion.orchestrator import ingest_one


def test_ingest_assigns_unique_ids_per_path_for_duplicate_files(tmp_path, monkeypatch):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    content = "duplicate"
    file_a.write_text(content)
    file_b.write_text(content)

    captured: dict[str, list[str]] = {}

    def fake_index_documents(chunks):
        for c in chunks:
            captured.setdefault(c["path"], []).append(c["id"])
        return len(chunks), []

    monkeypatch.setattr(
        "ingestion.io_loader.load_file_documents",
        lambda p: [Document(page_content=content, metadata={})],
    )
    monkeypatch.setattr(
        "ingestion.preprocess.preprocess_documents",
        lambda docs_like, normalized_path, ext: docs_like,
    )
    monkeypatch.setattr(
        "ingestion.preprocess.chunk_documents",
        lambda docs: [{"text": content}],
    )
    monkeypatch.setattr(
        "ingestion.storage.index_chunk_batch", fake_index_documents
    )
    def fake_index_chunks(chunks, os_index_batch=None):
        if os_index_batch:
            os_index_batch(chunks)
        return True

    monkeypatch.setattr(
        "utils.qdrant_utils.index_chunks_in_batches", fake_index_chunks
    )
    monkeypatch.setattr("ingestion.storage.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr(
        "ingestion.storage.is_duplicate_checksum", lambda c, p: False
    )
    monkeypatch.setattr("ingestion.storage.index_fulltext", lambda doc: None)

    ingest_one(str(file_a))
    ingest_one(str(file_b))

    assert len(captured) == 2
    paths = sorted(captured.keys())
    ids_a = set(captured[paths[0]])
    ids_b = set(captured[paths[1]])
    assert ids_a.isdisjoint(ids_b)

