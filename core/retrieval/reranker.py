from __future__ import annotations

from typing import Any, List, Optional, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    RETRIEVAL_ENABLE_RERANK,
    RETRIEVAL_RERANK_TIMEOUT_CONNECT,
    RETRIEVAL_RERANK_TIMEOUT_READ,
    RERANK_API_URL,
    logger,
)
from core.retrieval.types import DocHit


class CrossEncoderReranker:
    """
    Lightweight wrapper. Import is deferred so environments without
    sentence-transformers/torch can skip this feature.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; cross-encoder rerank is optional."
            ) from exc
        self.model: Any = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, docs: Sequence[DocHit], top_n: Optional[int] = None
    ) -> List[DocHit]:
        if not docs:
            return []
        docs_list = list(docs)
        pairs = [(query, d.get("text", "")) for d in docs_list]
        scores = list(self.model.predict(pairs).tolist())
        for d, s in zip(docs_list, scores):
            d["rerank_score"] = float(s)
        docs_list.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return docs_list[:top_n] if top_n else docs_list


class HttpCrossEncoderReranker:
    """
    Calls an external rerank service and maps scores back to retrieval docs.
    Endpoint contract:
      POST /rerank {"query": str, "documents": [str], "top_n": int?}
      -> {"scores": [float], "ranking": [int]}
    """

    def __init__(
        self,
        endpoint: str,
        *,
        connect_timeout_s: float = 2.0,
        read_timeout_s: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = (float(connect_timeout_s), float(read_timeout_s))
        self._session = session or requests.Session()
        if session is None:
            adapter = HTTPAdapter(
                pool_connections=16,
                pool_maxsize=16,
                max_retries=Retry(connect=2, read=0, total=0, backoff_factor=0.3),
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

    def rerank(
        self,
        query: str,
        docs: Sequence[DocHit],
        top_n: Optional[int] = None,
    ) -> List[DocHit]:
        if not docs:
            return []
        docs_list = list(docs)
        payload: dict[str, Any] = {
            "query": query,
            "documents": [str(doc.get("text", "") or "") for doc in docs_list],
        }
        if top_n is not None:
            payload["top_n"] = max(int(top_n), 1)

        response = self._session.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()

        raw_scores = body.get("scores")
        if isinstance(raw_scores, list):
            for doc, score in zip(docs_list, raw_scores):
                try:
                    doc["rerank_score"] = float(score)
                except (TypeError, ValueError):
                    continue

        ranked_docs: List[DocHit] = []
        raw_ranking = body.get("ranking")
        if isinstance(raw_ranking, list):
            seen: set[int] = set()
            for item in raw_ranking:
                try:
                    idx = int(item)
                except (TypeError, ValueError):
                    continue
                if idx < 0 or idx >= len(docs_list) or idx in seen:
                    continue
                ranked_docs.append(docs_list[idx])
                seen.add(idx)

        if not ranked_docs:
            ranked_docs = sorted(
                docs_list,
                key=lambda doc: float(doc.get("rerank_score", 0.0) or 0.0),
                reverse=True,
            )

        return ranked_docs[:top_n] if top_n else ranked_docs


def build_configured_reranker() -> HttpCrossEncoderReranker | None:
    if not RETRIEVAL_ENABLE_RERANK:
        return None
    if not RERANK_API_URL:
        logger.warning("Rerank is enabled but RERANK_API_URL is empty; skipping reranker.")
        return None
    return HttpCrossEncoderReranker(
        RERANK_API_URL,
        connect_timeout_s=RETRIEVAL_RERANK_TIMEOUT_CONNECT,
        read_timeout_s=RETRIEVAL_RERANK_TIMEOUT_READ,
    )
