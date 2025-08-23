import os
import sys
import types


# stub opensearchpy before importing module under test
class DummyOpenSearchException(Exception):
    pass


opensearchpy_stub = types.SimpleNamespace(
    OpenSearch=object,
    helpers=types.SimpleNamespace(),
    exceptions=types.SimpleNamespace(OpenSearchException=DummyOpenSearchException),
)
sys.modules.setdefault("opensearchpy", opensearchpy_stub)

import utils.opensearch_utils as osu


def test_ensure_ingest_log_index_exists(monkeypatch):
    class FakeIndices:
        def __init__(self):
            self.exists_called_with = None
            self.create_called_with = []

        def exists(self, index):
            self.exists_called_with = index
            return False

        def create(self, index, body, params=None):
            self.create_called_with.append((index, body))

    class FakeClient:
        def __init__(self):
            self.indices = FakeIndices()

    client = FakeClient()
    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: client)
    osu.ensure_ingest_log_index_exists()
    assert client.indices.exists_called_with == osu.INGEST_LOG_INDEX
    assert client.indices.create_called_with[0][0] == osu.INGEST_LOG_INDEX


def test_missing_indices(monkeypatch):
    class FakeIndices:
        def __init__(self, exists_map):
            self.exists_map = exists_map

        def exists(self, index):
            return self.exists_map.get(index, False)

    class FakeClient:
        def __init__(self, exists_map):
            self.indices = FakeIndices(exists_map)

    exists_map = {
        osu.OPENSEARCH_INDEX: False,
        osu.OPENSEARCH_FULLTEXT_INDEX: True,
        osu.INGEST_LOG_INDEX: False,
    }

    monkeypatch.setattr(
        "utils.opensearch_utils.get_client", lambda: FakeClient(exists_map)
    )
    missing = osu.missing_indices()
    assert missing == [osu.OPENSEARCH_INDEX, osu.INGEST_LOG_INDEX]


def test_list_files_from_opensearch(monkeypatch):
    class FakeClient:
        def search(self, index, body):
            return {
                "aggregations": {
                    "files": {
                        "buckets": [
                            {
                                "key": "dir/file1.txt",
                                "doc_count": 2,
                                "top_chunk": {
                                    "hits": {
                                        "hits": [
                                            {
                                                "_id": "1",
                                                "_source": {
                                                    "checksum": "c1",
                                                    "created_at": 1,
                                                    "modified_at": 2,
                                                    "indexed_at": 3,
                                                    "filetype": "txt",
                                                    "bytes": 100,
                                                    "size": "100 B",
                                                },
                                            }
                                        ]
                                    }
                                },
                            },
                            {
                                "key": "dir/file2.pdf",
                                "doc_count": 1,
                                "top_chunk": {
                                    "hits": {
                                        "hits": [
                                            {
                                                "_id": "2",
                                                "_source": {
                                                    "checksum": "c2",
                                                    "created_at": 4,
                                                    "modified_at": 5,
                                                    "indexed_at": 6,
                                                    "filetype": "pdf",
                                                    "bytes": 200,
                                                    "size": "200 B",
                                                },
                                            }
                                        ]
                                    }
                                },
                            },
                        ]
                    }
                }
            }

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    files = osu.list_files_from_opensearch(size=10)
    assert files[0]["path"] == "dir/file1.txt"
    assert files[0]["num_chunks"] == 2
    assert files[1]["filename"] == "file2.pdf"
    assert files[0]["bytes"] == 100
    assert files[1]["size"] == "200 B"


def test_get_duplicate_checksums_with_fallback(monkeypatch):
    class FakeClient:
        def search(self, index, body):
            field = body["aggs"]["by_checksum"]["aggs"]["distinct_paths"][
                "cardinality"
            ]["field"]
            if field == "path":
                raise Exception("no fielddata")
            return {
                "aggregations": {
                    "by_checksum": {
                        "buckets": [
                            {"key": "abc", "distinct_paths": {"value": 2}},
                            {"key": "def", "distinct_paths": {"value": 1}},
                        ]
                    }
                }
            }

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    dups = osu.get_duplicate_checksums()
    assert dups == ["abc"]


def test_is_duplicate_checksum(monkeypatch):
    class FakeClient:
        def count(self, index, body):
            must = body["query"]["bool"]["must"]
            must_not = body["query"]["bool"]["must_not"]
            if (
                must[0]["term"]["checksum"] == "c"
                and must_not[0]["term"]["path.keyword"] == "p1"
            ):
                return {"count": 1}
            return {"count": 0}

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    assert osu.is_duplicate_checksum("c", "p2") is False
    assert osu.is_duplicate_checksum("c", "p1") is True


def test_search_ingest_logs_builds_query(monkeypatch):
    recorded = {}

    class FakeClient:
        def search(self, index, body):
            recorded["index"] = index
            recorded["body"] = body
            return {"hits": {"hits": [{"_id": "1", "_source": {"status": "ok"}}]}}

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    res = osu.search_ingest_logs(
        status="ok", path_query="p", start="2020", end="2021", size=5
    )
    assert recorded["index"] == osu.INGEST_LOG_INDEX
    assert recorded["body"]["query"] == {
        "bool": {
            "must": [
                {"term": {"status": "ok"}},
                {"wildcard": {"path": "*p*"}},
            ],
            "filter": [{"range": {"attempt_at": {"gte": "2020", "lte": "2021"}}}],
        }
    }


def test_search_ingest_logs_handles_exception(monkeypatch):
    class FakeClient:
        def search(self, index, body):
            raise osu.exceptions.OpenSearchException("boom")

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    assert osu.search_ingest_logs(status="fail") == []
