import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

