from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence

DocHit = Dict[str, Any]


@dataclass
class RetrievalConfig:
    top_k: int = 5
    top_k_each: int = 20
    enable_variants: bool = True
    max_variants: int = 2
    variant_weights: Dict[str, float] = field(default_factory=dict)
    weight_exact_bm25: float = 1.0
    weight_rewrite_bm25: float = 0.5
    fusion_weight_vector: float = 0.7
    fusion_weight_bm25: float = 0.3
    enable_mmr: bool = True
    mmr_lambda: float = 0.6
    mmr_k: Optional[int] = None
    enable_rerank: bool = False
    rerank_top_n: int = 5
    sim_threshold: float = 0.90
    include_dups_if_needed: bool = True
    collapse_keep_limit: int = 64

    def with_top_k(self, top_k: int) -> "RetrievalConfig":
        return replace(self, top_k=top_k)


@dataclass
class RetrievalDeps:
    semantic_retriever: Callable[[str, int], Sequence[DocHit]]
    keyword_retriever: Callable[[str, int], Sequence[DocHit]]
    embed_texts: Optional[Callable[[List[str]], Any]] = None
    cross_encoder: Any = None


@dataclass
class RetrievalOutput:
    documents: List[DocHit]
    clarify: Optional[str] = None
