import importlib.util
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _EncodedVectors:
    def __init__(self, count: int):
        self._count = count

    def tolist(self):
        return [[float(i)] for i in range(self._count)]


class _Scores(list):
    def tolist(self):
        return list(self)


def _build_torch_stub():
    module = types.ModuleType("torch")
    module.float16 = "float16"
    module.inference_mode = lambda: _DummyContext()
    module.set_float32_matmul_precision = lambda *_args, **_kwargs: None
    module.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))
    )
    module.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda _idx: None,
        amp=types.SimpleNamespace(autocast=lambda **_kwargs: _DummyContext()),
    )
    return module


def _build_sentence_transformers_stub():
    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.model_name = model_name
            self._target_device = device

        def eval(self):
            return self

        def half(self):
            return self

        def encode(self, batch, **_kwargs):
            return _EncodedVectors(len(batch))

    class CrossEncoder:
        init_count = 0

        def __init__(self, model_name: str, device: str = "cpu"):
            self.model_name = model_name
            self.device = device
            CrossEncoder.init_count += 1

        def predict(self, pairs):
            # Deterministic increasing scores by document position.
            return _Scores(float(idx + 1) for idx, _ in enumerate(pairs))

    module.SentenceTransformer = SentenceTransformer
    module.CrossEncoder = CrossEncoder
    return module, CrossEncoder


def _load_embedder_app(monkeypatch):
    app_path = Path("embedder_api_multilingual/app.py").resolve()
    monkeypatch.syspath_prepend(str(app_path.parent))
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "stub-embedder-model")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "4")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("EMBEDDING_FP16", "false")
    monkeypatch.setenv("RERANK_MODEL_NAME", "stub-reranker-model")
    monkeypatch.setenv("RERANK_TOP_N_DEFAULT", "2")
    monkeypatch.setenv("RERANK_DEVICE", "cpu")

    monkeypatch.setitem(sys.modules, "torch", _build_torch_stub())
    sentence_transformers_stub, cross_encoder_cls = _build_sentence_transformers_stub()
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_stub)
    monkeypatch.delitem(sys.modules, "config", raising=False)

    module_name = "embedder_api_multilingual_app_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, cross_encoder_cls


def test_rerank_endpoint_returns_scores_and_top_ranking(monkeypatch):
    app_module, cross_encoder_cls = _load_embedder_app(monkeypatch)
    client = TestClient(app_module.app)

    response = client.post(
        "/rerank",
        json={
            "query": "what is the role",
            "documents": ["doc-a", "doc-b", "doc-c"],
            "top_n": 2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["scores"] == [1.0, 2.0, 3.0]
    assert payload["ranking"] == [2, 1]

    # Lazy load once and reuse the same reranker instance.
    response_2 = client.post(
        "/rerank",
        json={"query": "what is the role", "documents": ["x", "y"], "top_n": 1},
    )
    assert response_2.status_code == 200
    assert cross_encoder_cls.init_count == 1


def test_health_reports_rerank_settings_without_eager_load(monkeypatch):
    app_module, _cross_encoder_cls = _load_embedder_app(monkeypatch)
    client = TestClient(app_module.app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["rerank_model"] == "stub-reranker-model"
    assert payload["rerank_device"] == "cpu"
    assert payload["rerank_default_top_n"] == 2
    assert payload["rerank_loaded"] is False
