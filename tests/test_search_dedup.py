import sys
import os

from core import opensearch_store


def test_search_deduplicates_by_checksum(monkeypatch):
    class DummyClient:
        def search(self, index, body):
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "p1",
                            "_score": 1.0,
                            "_source": {"path": "a", "doc_id": "d1", "checksum": "abc"},
                            "inner_hits": {
                                "chunk": {
                                    "hits": {
                                        "hits": [
                                            {"_source": {"text": "t1", "chunk_index": 0}}
                                        ]
                                    }
                                }
                            },
                        },
                        {
                            "_id": "p2",
                            "_score": 0.9,
                            "_source": {"path": "b", "doc_id": "d2", "checksum": "abc"},
                            "inner_hits": {
                                "chunk": {
                                    "hits": {
                                        "hits": [
                                            {"_source": {"text": "t1", "chunk_index": 0}}
                                        ]
                                    }
                                }
                            },
                        },
                        {
                            "_id": "p3",
                            "_score": 0.8,
                            "_source": {"path": "c", "doc_id": "d3", "checksum": "def"},
                            "inner_hits": {
                                "chunk": {
                                    "hits": {
                                        "hits": [
                                            {"_source": {"text": "t2", "chunk_index": 0}}
                                        ]
                                    }
                                }
                            },
                        },
                    ]
                }
            }

    monkeypatch.setattr(opensearch_store, "get_client", lambda: DummyClient())
    results = opensearch_store.search("q", top_k=2)
    assert len(results) == 2
    checksums = [r["checksum"] for r in results]
    assert len(set(checksums)) == len(checksums)
