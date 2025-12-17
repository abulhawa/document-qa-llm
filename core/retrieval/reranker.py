from __future__ import annotations
from typing import List, Dict, Any, Optional


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
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self, query: str, docs: List[Dict[str, Any]], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = self.model.predict(pairs).tolist()
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return docs[:top_n] if top_n else docs
