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
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "1",
                        "_score": 1.0,
                        "_source": {
                            "path": "p",
                            "checksum": "c",
                            "text": "t",
                            "chunk_index": 0,
                            "modified_at": "2020",
                            "bytes": 123,
                            "size": "123 B",
                        },
                    },
                    {
                        "_id": "2",
                        "_score": 0.5,
                        "_source": {
                            "path": "p2",
                            "checksum": "c2",
                            "text": "t2",
                            "chunk_index": 1,
                            "modified_at": "2019",
                            "bytes": 456,
                            "size": "456 B",
                        },
                    },
                ]
            }
        }


def test_ensure_index_noop(monkeypatch):
    client = FakeClient()
    client.indices.exists_flag = True
    monkeypatch.setattr(osu, "get_client", lambda: client)
    osu.ensure_index_exists()
    assert client.indices.created == []


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
    assert files[0]["path"] == "p"
    assert files[0]["num_chunks"] == 1
    assert files[0]["bytes"] == 123
