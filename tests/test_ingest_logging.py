from core import ingestion
from langchain_core.documents import Document


class DummyClient:
    def __init__(self):
        self.docs = []

    def index(self, index: str, body: dict, **kwargs):
        self.docs.append(body)


def test_logs_emitted_for_up_to_date(monkeypatch, tmp_path):
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello")

    client = DummyClient()
    monkeypatch.setattr("utils.ingest_logging.get_client", lambda: client)
    monkeypatch.setattr(
        "utils.ingest_logging.ensure_ingest_log_index_exists", lambda: None
    )
    monkeypatch.setattr(
        "core.ingestion.is_file_up_to_date", lambda checksum, path: True
    )

    ingestion.ingest_one(str(file_path))

    assert client.docs, "expected a log document to be written"
    doc = client.docs[0]
    assert doc["status"] == "Already indexed"
    assert doc["bytes"] == 5
    assert doc["size"] == "5 B"


def test_duplicate_files_are_indexed_and_logged(monkeypatch, tmp_path):
    file_path = tmp_path / "dup.txt"
    file_path.write_text("hello")

    client = DummyClient()
    monkeypatch.setattr("utils.ingest_logging.get_client", lambda: client)
    monkeypatch.setattr(
        "utils.ingest_logging.ensure_ingest_log_index_exists", lambda: None
    )
    monkeypatch.setattr(
        "core.ingestion.is_file_up_to_date", lambda checksum, path: False
    )
    monkeypatch.setattr(
        "core.ingestion.is_duplicate_checksum", lambda checksum, path: True
    )
    monkeypatch.setattr(
        "core.ingestion.load_documents",
        lambda p: [Document(page_content="doc", metadata={})],
    )
    monkeypatch.setattr(
        "core.ingestion.preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: docs_like,
    )
    monkeypatch.setattr(
        "core.ingestion.split_documents", lambda docs: [{"text": "hello"}]
    )
    monkeypatch.setattr("core.ingestion.index_documents", lambda chunks: None)
    monkeypatch.setattr("utils.qdrant_utils.index_chunks_in_batches", lambda chunks: True)
    monkeypatch.setattr("core.ingestion.index_fulltext_document", lambda doc: None)

    result = ingestion.ingest_one(str(file_path))

    assert result["success"] is True
    assert result["status"] == "Duplicate & Indexed"
    assert client.docs and client.docs[0]["status"] == "Duplicate & Indexed"
