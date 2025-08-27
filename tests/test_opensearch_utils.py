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


def test_ensure_index_exists_creates_mapping(monkeypatch):
    created = {}

    class FakeIndices:
        def exists(self, index):
            return False

        def create(self, index, body, params=None):
            created["index"] = index
            created["body"] = body

    class FakeClient:
        indices = FakeIndices()

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    osu.ensure_index_exists()
    assert created["index"] == osu.OPENSEARCH_INDEX
    assert "mappings" in created["body"]


def test_index_documents_bulk(monkeypatch):
    recorded = {}

    class FakeClient:
        pass

    def fake_bulk(client, actions, **kwargs):
        recorded["actions"] = actions
        return (len(actions), [])

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    monkeypatch.setattr(osu, "helpers", types.SimpleNamespace(bulk=fake_bulk))
    monkeypatch.setattr(osu, "ensure_index_exists", lambda: None)

    chunks = [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]
    osu.index_documents(chunks)
    assert len(recorded["actions"]) == 2
    assert all(a.get("_op_type") == "create" for a in recorded["actions"])


def test_index_documents_update(monkeypatch):
    recorded = {}

    class FakeClient:
        pass

    def fake_bulk(client, actions, **kwargs):
        recorded["actions"] = actions
        return (len(actions), [])

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    monkeypatch.setattr(osu, "helpers", types.SimpleNamespace(bulk=fake_bulk))
    monkeypatch.setattr(osu, "ensure_index_exists", lambda: None)

    chunks = [{"id": "1", "text": "a", "op_type": "update"}]
    osu.index_documents(chunks)
    action = recorded["actions"][0]
    assert action["_op_type"] == "update"
    assert action["doc"]["text"] == "a"
