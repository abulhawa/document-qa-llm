import os
import types

import pytest
from langchain_core.documents import Document

from ingestion.doc_classifier import classify_document
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


def test_classify_document_detects_cv_metadata_from_filename():
    metadata = classify_document(
        "C:/docs/Jane_Doe_resume.pdf",
        "pdf",
        "Work experience\nEducation",
    )

    assert metadata["doc_type"] == "cv"
    assert metadata["doc_type_source"] == "rule"
    assert metadata["doc_type_confidence"] == pytest.approx(0.97)
    assert metadata["person_name"] == "Jane Doe"
    assert metadata["authority_rank"] == pytest.approx(1.0)


def test_classify_document_uses_text_name_when_filename_is_generic():
    metadata = classify_document(
        "C:/docs/resume.pdf",
        "pdf",
        "John Doe\nCurriculum Vitae\nExperience",
    )

    assert metadata["doc_type"] == "cv"
    assert metadata["doc_type_source"] == "rule"
    assert metadata["doc_type_confidence"] == pytest.approx(0.97)
    assert metadata["person_name"] == "John Doe"
    assert metadata["authority_rank"] == pytest.approx(1.0)


def test_classify_document_detects_non_identity_doc_type_from_filename():
    metadata = classify_document(
        "C:/docs/insurance_premium_adjustment_2025.pdf",
        "pdf",
        "Dear policyholder, your premium changes next month.",
    )

    assert metadata["doc_type"] == "insurance_letter"
    assert metadata["doc_type_source"] == "rule"
    assert metadata["doc_type_confidence"] == pytest.approx(0.95)
    assert metadata["person_name"] is None
    assert metadata["authority_rank"] is None


def test_classify_document_falls_back_to_other_when_unmatched():
    metadata = classify_document(
        "C:/docs/random_notes_blob.txt",
        "txt",
        "Quick scratch notes without any known document structure.",
    )

    assert metadata["doc_type"] == "other"
    assert metadata["doc_type_source"] == "fallback"
    assert metadata["doc_type_confidence"] == pytest.approx(0.25)
    assert metadata["person_name"] is None
    assert metadata["authority_rank"] is None


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


def test_ingest_one_persists_classifier_metadata(tmp_path, monkeypatch):
    f = tmp_path / "john_doe_cv.txt"
    f.write_text("hello")

    monkeypatch.setattr("utils.file_utils.compute_checksum", lambda p: "abc")
    monkeypatch.setattr("ingestion.storage.is_file_up_to_date", lambda c, p: False)
    monkeypatch.setattr("ingestion.storage.is_duplicate_checksum", lambda c, p: False)
    monkeypatch.setattr("ingestion.orchestrator.IngestLogEmitter", DummyLog)
    monkeypatch.setattr(
        "ingestion.orchestrator.classify_document",
        lambda path, filetype, full_text: {
            "doc_type": "cv",
            "doc_type_confidence": 0.97,
            "doc_type_source": "rule",
            "person_name": "John Doe",
            "authority_rank": 1.0,
        },
    )

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

    captured = {}

    def fake_index_documents(chunks):
        captured["chunk"] = chunks[0]
        return len(chunks), []

    def fake_index_fulltext(doc):
        captured["fulltext"] = doc

    monkeypatch.setattr("ingestion.storage.index_chunk_batch", fake_index_documents)

    def fake_index_chunks(chunks, os_index_batch=None):
        if os_index_batch:
            os_index_batch(chunks)
        return True

    monkeypatch.setattr(
        "utils.qdrant_utils.index_chunks_in_batches", fake_index_chunks
    )
    monkeypatch.setattr("ingestion.storage.index_fulltext", fake_index_fulltext)

    result = ingest_one(str(f))

    assert result["success"] is True
    assert captured["chunk"]["doc_type"] == "cv"
    assert captured["chunk"]["doc_type_confidence"] == pytest.approx(0.97)
    assert captured["chunk"]["doc_type_source"] == "rule"
    assert captured["chunk"]["person_name"] == "John Doe"
    assert captured["chunk"]["authority_rank"] == pytest.approx(1.0)
    assert captured["fulltext"]["doc_type"] == "cv"
    assert captured["fulltext"]["doc_type_confidence"] == pytest.approx(0.97)
    assert captured["fulltext"]["doc_type_source"] == "rule"
    assert captured["fulltext"]["person_name"] == "John Doe"
    assert captured["fulltext"]["authority_rank"] == pytest.approx(1.0)
