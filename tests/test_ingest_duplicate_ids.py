import os
import uuid
from langchain_core.documents import Document

from ingestion.orchestrator import ingest_one


def test_ingest_assigns_unique_ids_per_path_for_duplicate_files(tmp_path, monkeypatch):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    content = "duplicate"
    file_a.write_text(content)
    file_b.write_text(content)

    captured: dict[str, list[str]] = {}
    stored_fulltext: dict[str, dict] = {}

    def fake_index_documents(chunks):
        for c in chunks:
            captured.setdefault(c["path"], []).append(c["id"])
        return len(chunks), []

    monkeypatch.setattr("ingestion.io_loader.compute_checksum", lambda p: "abc123")
    monkeypatch.setattr("utils.file_utils.get_file_size", lambda p: len(content))
    monkeypatch.setattr(
        "utils.file_utils.get_file_timestamps",
        lambda p: {"created": "2023-01-01", "modified": "2023-01-01"},
    )
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
    monkeypatch.setattr("ingestion.storage.index_chunk_batch", fake_index_documents)

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
    monkeypatch.setattr(
        "ingestion.storage.index_fulltext",
        lambda doc: stored_fulltext.setdefault(doc["checksum"], doc),
    )
    monkeypatch.setattr(
        "ingestion.storage.get_existing_fulltext",
        lambda checksum: stored_fulltext.get(checksum),
    )
    monkeypatch.setattr(
        "ingestion.storage.get_fulltext_for_path", lambda path: None
    )

    ingest_one(str(file_a))
    ingest_one(str(file_b))

    assert len(captured) == 1
    canonical_path = next(iter(captured.keys()))
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "abc123:0"))
    assert set(captured[canonical_path]) == {expected_id}
    full_doc = stored_fulltext["abc123"]
    assert os.path.normpath(full_doc["path"]) == os.path.normpath(canonical_path)
    normalized_aliases = {os.path.normpath(p) for p in full_doc["aliases"]}
    expected_aliases = {os.path.normpath(str(file_a)), os.path.normpath(str(file_b))}
    expected_aliases.discard(os.path.normpath(canonical_path))
    assert normalized_aliases == expected_aliases
    assert os.path.normpath(canonical_path) not in normalized_aliases
