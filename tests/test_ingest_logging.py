from ingestion import orchestrator
from ingestion import io_loader, preprocess, storage
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
        "ingestion.storage.is_file_up_to_date", lambda checksum, path: True
    )

    orchestrator.ingest_one(str(file_path))

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
        "ingestion.storage.is_file_up_to_date", lambda checksum, path: False
    )
    monkeypatch.setattr(
        "ingestion.storage.is_duplicate_checksum", lambda checksum, path: True
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
    monkeypatch.setattr(
        "ingestion.storage.index_chunk_batch", lambda chunks: (len(chunks), [])
    )
    monkeypatch.setattr(
        "ingestion.storage.embed_and_store",
        lambda chunks, os_index_batch=None: True,
    )
    monkeypatch.setattr("ingestion.storage.index_fulltext", lambda doc: None)

    result = orchestrator.ingest_one(str(file_path))

    assert result["success"] is True
    assert result["status"] == "Duplicate & Indexed"
    assert client.docs and client.docs[0]["status"] == "Duplicate & Indexed"
