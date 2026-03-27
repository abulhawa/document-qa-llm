import pytest

from core.retrieval.reranker import HttpCrossEncoderReranker


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _DummySession:
    def __init__(self, payload):
        self.payload = payload
        self.last_post = None

    def post(self, url, json=None, timeout=None):
        self.last_post = {"url": url, "json": json, "timeout": timeout}
        return _DummyResponse(self.payload)


def test_http_reranker_orders_by_service_ranking_and_sets_scores():
    session = _DummySession({"scores": [0.1, 0.8, 0.2], "ranking": [1, 2]})
    reranker = HttpCrossEncoderReranker(
        "http://localhost:8000/rerank",
        connect_timeout_s=1.5,
        read_timeout_s=9.0,
        session=session,
    )
    docs = [
        {"id": "d1", "text": "first"},
        {"id": "d2", "text": "second"},
        {"id": "d3", "text": "third"},
    ]

    reranked = reranker.rerank("q", docs, top_n=2)

    assert session.last_post is not None
    assert session.last_post["url"] == "http://localhost:8000/rerank"
    assert session.last_post["json"] == {
        "query": "q",
        "documents": ["first", "second", "third"],
        "top_n": 2,
    }
    assert session.last_post["timeout"] == pytest.approx((1.5, 9.0))
    assert [doc.get("id") for doc in reranked] == ["d2", "d3"]
    assert docs[0]["rerank_score"] == pytest.approx(0.1)
    assert docs[1]["rerank_score"] == pytest.approx(0.8)
    assert docs[2]["rerank_score"] == pytest.approx(0.2)


def test_http_reranker_falls_back_to_score_sort_when_ranking_missing():
    session = _DummySession({"scores": [-2.5, -1.0, -3.0]})
    reranker = HttpCrossEncoderReranker(
        "http://localhost:8000/rerank",
        session=session,
    )
    docs = [
        {"id": "d1", "text": "first"},
        {"id": "d2", "text": "second"},
        {"id": "d3", "text": "third"},
    ]

    reranked = reranker.rerank("q", docs, top_n=2)

    assert [doc.get("id") for doc in reranked] == ["d2", "d1"]
