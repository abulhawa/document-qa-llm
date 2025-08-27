import os
from langchain_core.documents import Document

from core.ingestion import ingest_one


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

    monkeypatch.setattr(
        "core.ingestion.load_documents",
        lambda p: [Document(page_content=content, metadata={})],
    )
    monkeypatch.setattr(
        "core.ingestion.preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: docs_like,
    )
    monkeypatch.setattr(
        "core.ingestion.split_documents", lambda docs: [{"text": content}]
    )
    monkeypatch.setattr("core.ingestion.index_documents", fake_index_documents)
    monkeypatch.setattr("utils.qdrant_utils.index_chunks", lambda chunks: True)
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr(
        "core.ingestion.is_duplicate_checksum", lambda c, p: False
    )
    monkeypatch.setattr("core.ingestion.index_fulltext_document", lambda doc: None)

    ingest_one(str(file_a))
    ingest_one(str(file_b))

    assert len(captured) == 2
    paths = sorted(captured.keys())
    ids_a = set(captured[paths[0]])
    ids_b = set(captured[paths[1]])
    assert ids_a.isdisjoint(ids_b)

