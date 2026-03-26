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
    anchored_exact_only: bool = True
    variant_weights: Dict[str, float] = field(default_factory=dict)
    weight_exact_bm25: float = 1.0
    weight_rewrite_bm25: float = 0.5
    fusion_weight_vector: float = 0.7
    fusion_weight_bm25: float = 0.3
    anchored_lexical_bias_enabled: bool = True
    anchored_fusion_weight_vector: float = 0.4
    anchored_fusion_weight_bm25: float = 0.6
    enable_mmr: bool = True
    mmr_lambda: float = 0.6
    mmr_k: Optional[int] = None
    enable_rerank: bool = False
    rerank_top_n: int = 5
    sim_threshold: float = 0.82
    include_dups_if_needed: bool = True
    collapse_keep_limit: int = 64
    authority_boost_enabled: bool = True
    authority_boost_weight: float = 0.08
    authority_boost_max_fraction: float = 0.15
    recency_boost_enabled: bool = True
    recency_boost_weight: float = 0.06
    recency_boost_half_life_days: float = 120.0
    recency_boost_max_fraction: float = 0.12
    cv_family_collapse_enabled: bool = True
    cv_family_relevance_margin: float = 0.10
    profile_intent_boost_enabled: bool = True
    profile_intent_boost_weight: float = 0.10
    profile_intent_boost_max_fraction: float = 0.20
    abstention_enabled: bool = True
    abstention_min_overlap_terms: int = 2

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
