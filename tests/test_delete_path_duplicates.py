import sys
import os
import types

# Stub external dependencies before importing project modules
opensearchpy_stub = types.SimpleNamespace(
    exceptions=types.SimpleNamespace(OpenSearchException=Exception),
    helpers=types.SimpleNamespace(),
    OpenSearch=object,
)
sys.modules.setdefault("opensearchpy", opensearchpy_stub)

qdrant_models_stub = types.SimpleNamespace(
    Filter=lambda **kwargs: types.SimpleNamespace(**kwargs),
    FieldCondition=lambda **kwargs: types.SimpleNamespace(**kwargs),
    MatchValue=lambda **kwargs: types.SimpleNamespace(**kwargs),
    MatchAny=lambda **kwargs: types.SimpleNamespace(**kwargs),
    FilterSelector=lambda **kwargs: types.SimpleNamespace(**kwargs),
    PointStruct=object,
    VectorParams=object,
    Distance=types.SimpleNamespace(COSINE=0),
)
class DummyQdrantClientStub:
    def __init__(self, *args, **kwargs):
        pass

qdrant_client_stub = types.SimpleNamespace(
    QdrantClient=DummyQdrantClientStub, models=qdrant_models_stub
)
sys.modules.setdefault("qdrant_client", qdrant_client_stub)
sys.modules.setdefault("qdrant_client.http", types.SimpleNamespace(models=qdrant_models_stub))
sys.modules.setdefault("qdrant_client.http.models", qdrant_models_stub)
sys.modules.setdefault("requests", types.SimpleNamespace())

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import opensearch_utils, qdrant_utils


def test_delete_single_path_leaves_other_duplicate(monkeypatch):
    class DummyOSClient:
        def __init__(self):
            self.docs = [
                {"path": "a/foo.txt", "checksum": "abc"},
                {"path": "b/foo.txt", "checksum": "abc"},
            ]

        def delete_by_query(self, index, body, params):
            should = body["query"]["bool"]["should"]
            deleted = 0
            for clause in should:
                must = clause["bool"]["must"]
                term_path = must[0]["term"]
                path = term_path.get("path.keyword") or term_path.get("path")
                checksum = must[1]["term"]["checksum"]
                before = len(self.docs)
                self.docs = [
                    d for d in self.docs if not (d["path"] == path and d["checksum"] == checksum)
                ]
                deleted += before - len(self.docs)
            return {"deleted": deleted}

    class DummyQdrantClient:
        def __init__(self):
            self.docs = [
                {"path": "a/foo.txt", "checksum": "abc"},
                {"path": "b/foo.txt", "checksum": "abc"},
            ]

        def delete(self, collection_name, points_selector):
            flt = points_selector.filter
            path = None
            checksum = None
            for cond in flt.must:
                if cond.key == "path":
                    path = cond.match.value
                elif cond.key == "checksum":
                    checksum = cond.match.value
            self.docs = [
                d for d in self.docs if not (d["path"] == path and d["checksum"] == checksum)
            ]

    dummy_os = DummyOSClient()
    dummy_qdrant = DummyQdrantClient()
    monkeypatch.setattr(opensearch_utils, "get_client", lambda: dummy_os)
    monkeypatch.setattr(qdrant_utils, "client", dummy_qdrant)

    pairs = [("b/foo.txt", "abc")]
    deleted = opensearch_utils.delete_files_by_path_checksum(pairs)
    qdrant_utils.delete_vectors_by_path_checksum(pairs)

    assert deleted == 1
    assert dummy_os.docs == [{"path": "a/foo.txt", "checksum": "abc"}]
    assert dummy_qdrant.docs == [{"path": "a/foo.txt", "checksum": "abc"}]
