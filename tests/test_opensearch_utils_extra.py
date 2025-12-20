import types
import logging
import utils.opensearch_utils as osu


class FakeIndices:
    def __init__(self, exists=False):
        self.exists_flag = exists
        self.created = []

    def exists(self, index):
        return self.exists_flag

    def create(self, index, body, params=None):
        self.created.append((index, body))


class FakeClient:
    def __init__(self):
        self.indices = FakeIndices()
        self.bulk_ops = []
        self.deleted = []
        self.search_calls = []

    def bulk(self, body, params):
        self.bulk_ops.append(body)
        items = []
        for op, doc in zip(body[::2], body[1::2]):
            if doc.get("doc", {}).get("fail"):
                items.append({"update": {"error": "x"}})
            else:
                items.append({"update": {"result": "updated"}})
        return {"items": items}

    def delete_by_query(self, index, body, params):
        self.deleted.append((index, body))
        q = body["query"]
        if "terms" in q:
            return {"deleted": len(q["terms"]["checksum"])}
        return {"deleted": len(q["bool"]["should"])}

    def search(self, index, body):
        self.search_calls.append({"index": index, "body": body})
        if index == osu.CHUNKS_INDEX:
            return {
                "aggregations": {
                    "by_path": {
                        "buckets": [
                            {
                                "key": "p",
                                "doc_count": 1,
                                "sample": {
                                    "hits": {
                                        "hits": [
                                            {
                                                "_source": {
                                                    "path": "p",
                                                    "checksum": "c",
                                                    "text": "t",
                                                    "chunk_index": 0,
                                                    "modified_at": "2020",
                                                    "created_at": "2019",
                                                    "indexed_at": "2021",
                                                    "bytes": 123,
                                                    "size": "123 B",
                                                    "filetype": "txt",
                                                }
                                            }
                                        ]
                                    }
                                },
                            }
                        ]
                    }
                }
            }
        return {"hits": {"hits": []}}

    def get(self, index, id):
        return {
            "_id": id,
            "_source": {
                "path": "p",
                "aliases": ["p2", "p3"],
                "filetype": "txt",
                "created_at": "2019",
                "modified_at": "2020",
                "indexed_at": "2021",
                "size_bytes": 123,
                "checksum": id,
            },
        }


def test_bulk_index_partial_failure(monkeypatch, caplog):
    client = FakeClient()
    monkeypatch.setattr(osu, "get_client", lambda: client)

    def fake_bulk(client, actions, **kwargs):
        return (1, ["err"])

    monkeypatch.setattr(osu, "helpers", types.SimpleNamespace(bulk=fake_bulk))
    caplog.set_level(logging.ERROR)
    osu.index_documents([{"id": "1", "text": "a"}])
    assert any("OpenSearch indexing failed" in r.message for r in caplog.records)


def test_get_files_by_checksum(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(osu, "get_client", lambda: client)
    files = osu.get_files_by_checksum("c")
    assert {f["path"] for f in files} == {"p", "p2", "p3"}
    canonical = [f for f in files if f["location_type"] == "canonical"][0]
    assert canonical["path"] == "p"
    assert canonical["num_chunks"] == 1
    assert canonical["bytes"] == 123
    alias_paths = {f["path"] for f in files if f["location_type"] == "alias"}
    assert alias_paths == {"p2", "p3"}
