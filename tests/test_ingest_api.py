import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ingestion import ingest


def test_ingest_accepts_single_path(tmp_path, monkeypatch):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello")

    # Stub out external dependencies used during ingestion
    monkeypatch.setattr("core.ingestion.index_documents", lambda chunks: None)
    monkeypatch.setattr("utils.qdrant_utils.index_chunks", lambda chunks: True)
    class DummyApp:
        def send_task(self, *args, **kwargs):
            pass
    monkeypatch.setattr("core.ingestion.celery_app", DummyApp())
    monkeypatch.setattr(
        "core.ingestion.is_file_up_to_date", lambda checksum, path: False
    )

    result_str = ingest(str(sample))
    result_list = ingest([str(sample)])

    assert result_str == result_list
    assert result_str and result_str[0]["path"] == os.path.normpath(str(sample)).replace("\\", "/")
