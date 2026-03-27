import sys
import os

from core import opensearch_store


def test_search_deduplicates_by_checksum(monkeypatch):
    class DummyClient:
        def search(self, index, body):
            return {
                "hits": {
                    "hits": [
                        {"_id": "1", "_score": 1.0, "_source": {"path": "a", "text": "t1", "chunk_index": 0, "modified_at": "", "checksum": "abc"}},
                        {"_id": "2", "_score": 0.9, "_source": {"path": "b", "text": "t1", "chunk_index": 0, "modified_at": "", "checksum": "abc"}},
                        {"_id": "3", "_score": 0.8, "_source": {"path": "c", "text": "t2", "chunk_index": 0, "modified_at": "", "checksum": "def"}},
                    ]
                }
            }

    monkeypatch.setattr(opensearch_store, "get_client", lambda: DummyClient())
    results = opensearch_store.search("q", top_k=2)
    assert len(results) == 2
    checksums = [r["checksum"] for r in results]
    assert len(set(checksums)) == len(checksums)


def test_search_queries_text_path_and_filename_fields(monkeypatch):
    captured = {}

    class DummyClient:
        def search(self, index, body):
            captured["index"] = index
            captured["body"] = body
            return {"hits": {"hits": []}}

    monkeypatch.setattr(opensearch_store, "get_client", lambda: DummyClient())
    opensearch_store.search("ali cv contact", top_k=3)

    query = captured["body"]["query"]["multi_match"]
    assert query["query"] == "ali cv contact"
    assert query["type"] == "best_fields"
    assert query["operator"] == "or"
    assert query["fields"] == [
        "text^1.0",
        "path^0.35",
        "filename^0.75",
        "filename.keyword^1.10",
    ]
