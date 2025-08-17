import os
import sys
import types


# Stub opensearchpy before importing module under test
opensearchpy_stub = types.SimpleNamespace(OpenSearch=object, helpers=types.SimpleNamespace(), exceptions=Exception)
sys.modules.setdefault("opensearchpy", opensearchpy_stub)

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

        def create(self, index, body):
            created["index"] = index
            created["body"] = body

    class FakeClient:
        indices = FakeIndices()

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    osu.ensure_index_exists()
    assert created["index"] == osu.OPENSEARCH_INDEX
    props = created["body"]["mappings"]["properties"]
    assert "relation" in props and props["relation"]["type"] == "join"



def test_index_documents_bulk(monkeypatch):
    recorded = {}

    class FakeClient:
        pass

    def fake_bulk(client, actions, **kw):
        recorded["actions"] = actions
        return (len(actions), [])

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())
    monkeypatch.setattr(osu, "helpers", types.SimpleNamespace(bulk=fake_bulk))
    monkeypatch.setattr(osu, "ensure_index_exists", lambda: None)

    chunks = [
        {"id": "1", "text": "a", "doc_id": "d", "filename": "f", "path": "p"},
        {"id": "2", "text": "b", "doc_id": "d", "filename": "f", "path": "p"},
    ]
    osu.index_documents(chunks)
    # one parent + two children
    assert len(recorded["actions"]) == 3
    assert recorded["actions"][0]["_id"] == "d"



def test_set_has_embedding_true_by_ids(monkeypatch):
    class FakeClient:
        def bulk(self, body, params):
            # simulate success for each update
            items = []
            for i in range(0, len(body), 2):
                items.append({"update": {"result": "updated"}})
            return {"items": items}

    monkeypatch.setattr("utils.opensearch_utils.get_client", lambda: FakeClient())

    updated, errors = osu.set_has_embedding_true_by_ids(["a", "b", "a"])
    assert updated == 2
    assert errors == 0
