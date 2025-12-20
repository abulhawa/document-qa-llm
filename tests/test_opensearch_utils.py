import os
import types
import utils.opensearch_utils as osu


def test_is_file_up_to_date_does_not_match_similar_paths(monkeypatch):
    existing_path = "/foo/bar-baz.txt"
    checksum = "123"

    class FakeClient:
        def count(self, index, body):
            must = body["query"]["bool"]["must"]
            path_clause = must[1]
            # if query uses match_phrase, simulate partial match allowing prefix
            if "match_phrase" in path_clause:
                query_path = path_clause["match_phrase"]["path"]
                if existing_path.startswith(query_path):
                    return {"count": 1}
                return {"count": 0}
            # otherwise expect term on path.keyword
            query_path = path_clause["term"].get("path.keyword")
            if query_path == existing_path:
                return {"count": 1}
            return {"count": 0}

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())

    # exact path should be considered up to date
    assert osu.is_file_up_to_date(checksum, existing_path) is True
    # similar path should not be considered up to date
    assert osu.is_file_up_to_date(checksum, "/foo/bar.txt") is False


def test_index_documents_bulk(monkeypatch):
    recorded = {}

    class FakeClient:
        pass

    def fake_bulk(client, actions, **kwargs):
        recorded["actions"] = actions
        return (len(actions), [])

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    monkeypatch.setattr(osu, "helpers", types.SimpleNamespace(bulk=fake_bulk))

    chunks = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
    osu.index_documents(chunks)
    assert len(recorded["actions"]) == 2
    assert all(a.get("_op_type") == "index" for a in recorded["actions"])


def test_index_documents_update(monkeypatch):
    recorded = {}

    class FakeClient:
        pass

    def fake_bulk(client, actions, **kwargs):
        recorded["actions"] = actions
        return (len(actions), [])

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    monkeypatch.setattr(osu, "helpers", types.SimpleNamespace(bulk=fake_bulk))

    chunks = [{"id": "1", "text": "a", "op_type": "update"}]
    osu.index_documents(chunks)
    action = recorded["actions"][0]
    assert action["_op_type"] == "update"
    assert action["doc"]["text"] == "a"


def test_get_fulltext_by_path_or_alias(monkeypatch):
    captured = {}

    class FakeClient:
        def search(self, *, index, body):
            captured["body"] = body
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "doc1",
                            "_source": {
                                "path": "/foo/bar.txt",
                                "aliases": ["/other/location.txt"],
                                "checksum": "abc123",
                            },
                        }
                    ]
                }
            }

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    doc = osu.get_fulltext_by_path_or_alias("/other/location.txt")

    assert doc["id"] == "doc1"
    assert doc["path"] == "/foo/bar.txt"
    assert captured["body"]["query"]["bool"]["should"][1]["term"] == {
        "aliases": "/other/location.txt"
    }
