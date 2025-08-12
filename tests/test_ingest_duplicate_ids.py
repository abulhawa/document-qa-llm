import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ingestion import ingest


def test_ingest_assigns_unique_ids_per_path_for_duplicate_files(tmp_path, monkeypatch):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    content = "duplicate"
    file_a.write_text(content)
    file_b.write_text(content)

    captured = {}

    def fake_index_documents(chunks):
        for c in chunks:
            captured.setdefault(c["path"], []).append(c["id"])

    monkeypatch.setattr("core.ingestion.index_documents", fake_index_documents)
    monkeypatch.setattr("utils.qdrant_utils.index_chunks", lambda chunks: True)

    class DummyApp:
        def send_task(self, *args, **kwargs):
            pass

    monkeypatch.setattr("core.ingestion.celery_app", DummyApp())
    monkeypatch.setattr("core.ingestion.is_file_up_to_date", lambda checksum, path: False)
    monkeypatch.setattr("core.ingestion.MAX_WORKERS", 1)

    ingest([str(file_a), str(file_b)])

    assert len(captured) == 2
    paths = sorted(captured.keys())
    ids_a = set(captured[paths[0]])
    ids_b = set(captured[paths[1]])
    assert ids_a.isdisjoint(ids_b)

