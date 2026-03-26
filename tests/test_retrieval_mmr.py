from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List


def _load_real_mmr_select():
    module_path = Path(__file__).resolve().parents[1] / "core" / "retrieval" / "mmr.py"
    spec = importlib.util.spec_from_file_location("retrieval_mmr_real", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mmr_select


def _build_embed_lookup(vectors: Dict[str, List[float]]):
    def _embed(texts: List[str]) -> List[List[float]]:
        return [vectors[text] for text in texts]

    return _embed


def test_score_aware_mmr_keeps_high_ranked_doc_over_low_rank_novel_doc():
    mmr_select = _load_real_mmr_select()
    vectors = {
        "query": [1.0, 0.0],
        "doc_a": [1.0, 0.0],
        "doc_b": [1.0, 0.0],
        "doc_c": [0.5, 0.8660254],
    }
    docs = [
        {"id": "a", "text": "doc_a", "retrieval_score": 1.0},
        {"id": "b", "text": "doc_b", "retrieval_score": 0.95},
        {"id": "c", "text": "doc_c", "retrieval_score": 0.1},
    ]

    selected = mmr_select(
        "query",
        docs,
        embed=_build_embed_lookup(vectors),
        k=2,
        lambda_mult=0.6,
        retrieval_weight=0.5,
    )

    assert [doc.get("id") for doc in selected] == ["a", "b"]


def test_score_aware_mmr_still_diversifies_when_alternative_is_strong():
    mmr_select = _load_real_mmr_select()
    vectors = {
        "query": [1.0, 0.0],
        "doc_a": [1.0, 0.0],
        "doc_b": [1.0, 0.0],
        "doc_c": [0.7, 0.7141428],
    }
    docs = [
        {"id": "a", "text": "doc_a", "retrieval_score": 1.0},
        {"id": "b", "text": "doc_b", "retrieval_score": 0.5},
        {"id": "c", "text": "doc_c", "retrieval_score": 0.9},
    ]

    selected = mmr_select(
        "query",
        docs,
        embed=_build_embed_lookup(vectors),
        k=2,
        lambda_mult=0.6,
        retrieval_weight=0.5,
    )

    assert [doc.get("id") for doc in selected] == ["a", "c"]
