import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
