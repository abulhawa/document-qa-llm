from core import ingestion


class DummyClient:
    def __init__(self):
        self.docs = []

    def index(self, index: str, id: str, document: dict):
        self.docs.append(document)


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
    assert client.docs[0]["status"] == "skipped_up_to_date"
