from typing import List, Optional, Tuple, Sequence

from config import logger

from core.embeddings import embed_texts
from core.vector_store import retrieve_top_k as semantic_retriever
from core.opensearch_store import search as keyword_retriever

from core.retrieval.fusion import fuse_semantic_and_bm25, dedup_by_checksum
from core.retrieval.variants import generate_variants
from core.retrieval.mmr import mmr_select
from core.retrieval.dedup import collapse_near_duplicates
from core.retrieval.reranker import CrossEncoderReranker
from core.retrieval.types import RetrievalConfig, RetrievalDeps, RetrievalOutput, DocHit


def _default_deps() -> RetrievalDeps:
    return RetrievalDeps(
        semantic_retriever=semantic_retriever,
        keyword_retriever=keyword_retriever,
        embed_texts=embed_texts,
        cross_encoder=None,
    )


def _apply_variant_weights(
    variants_with_weights: Sequence[Tuple[str, float]],
    cfg: RetrievalConfig,
) -> List[Tuple[str, float]]:
    weighted: List[Tuple[str, float]] = []
    for idx, (text, weight) in enumerate(variants_with_weights):
        if idx == 0:
            default_weight = cfg.variant_weights.get("exact", cfg.weight_exact_bm25)
        else:
            default_weight = cfg.variant_weights.get("rewrite", cfg.weight_rewrite_bm25)
        weighted.append((text, weight if weight is not None else default_weight))
    return weighted


def retrieve(
    query: str,
    *,
    cfg: RetrievalConfig,
    deps: Optional[RetrievalDeps] = None,
) -> RetrievalOutput:
    """
    Configurable retrieval pipeline combining keyword + semantic search with
    optional variants, fusion, near-duplicate collapsing, MMR, and reranking.
    """

    deps = deps or _default_deps()
    logger.info(f"Running retrieval: '{query}'")

    # 1) Generate variants (keep rewritten + exact) if enabled
    clarify_msg: Optional[str] = None
    variants_with_weights: List[Tuple[str, float]] = []
    if cfg.enable_variants:
        variants_result = generate_variants(query)
        if "clarify" in variants_result:
            clarify_msg = variants_result["clarify"]
            return RetrievalOutput(documents=[], clarify=clarify_msg)
        variants_with_weights = variants_result.get("variants", [])
    if not variants_with_weights:
        variants_with_weights = [(query.strip(), cfg.weight_exact_bm25)]

    if cfg.max_variants > 0:
        variants_with_weights = variants_with_weights[: cfg.max_variants]

    variants_with_weights = _apply_variant_weights(variants_with_weights, cfg)

    # 2) Retrieve per variant and collect with weights
    vector_results_all: List[DocHit] = []
    bm25_results_all: List[DocHit] = []

    for idx, (qv, weight_override) in enumerate(variants_with_weights):
        v = deps.semantic_retriever(qv, cfg.top_k_each)
        vector_results_all.extend(v)

        b = deps.keyword_retriever(qv, cfg.top_k_each)
        for hit in b:
            hit["_bm25_variant_weight"] = weight_override
            hit["_variant_rank"] = idx
        bm25_results_all.extend(b)

    # 3) Fuse with normalization (respect variant weights)
    fused = fuse_semantic_and_bm25(
        vector_results_all,
        bm25_results_all,
        w_vec=cfg.fusion_weight_vector,
        w_bm25=cfg.fusion_weight_bm25,
    )
    for d in fused:
        w = d.get("_bm25_variant_weight", 1.0)
        d["retrieval_score"] = cfg.fusion_weight_vector * d.get("score_vector", 0.0) + cfg.fusion_weight_bm25 * (
            d.get("score_bm25", 0.0) * w
        )

    fused_sorted: List[DocHit] = sorted(
        fused,
        key=lambda x: (x.get("retrieval_score", 0.0), x.get("modified_at", "")),
        reverse=True,
    )
    unique_docs = dedup_by_checksum(fused_sorted)

    # 4) Near-duplicate collapse (keep one rep; stage dups aside)
    kept, dups = (unique_docs, [])
    if deps.embed_texts is not None and cfg.sim_threshold > 0:
        kept, dups = collapse_near_duplicates(
            unique_docs,
            embed_texts=deps.embed_texts,
            sim_threshold=cfg.sim_threshold,
            keep_limit=cfg.collapse_keep_limit,
        )

    # 5) MMR (optional) for diversity
    docs_for_answer: List[DocHit]
    if cfg.enable_mmr and deps.embed_texts is not None:
        docs_for_answer = mmr_select(
            query,
            kept,
            embed=deps.embed_texts,
            k=min(cfg.mmr_k or cfg.top_k, cfg.top_k),
            lambda_mult=cfg.mmr_lambda,
        )
    else:
        docs_for_answer = kept[: cfg.top_k]

    # If short, optionally top-up from duplicates
    if cfg.include_dups_if_needed and len(docs_for_answer) < cfg.top_k and dups:
        need = cfg.top_k - len(docs_for_answer)
        docs_for_answer.extend(dups[:need])

    # Optional cross-encoder rerank
    if cfg.enable_rerank and deps.cross_encoder is not None and docs_for_answer:
        docs_for_answer = deps.cross_encoder.rerank(
            query, docs_for_answer, top_n=min(cfg.rerank_top_n, len(docs_for_answer))
        )

    logger.info(
        "Retrieval returned %s results after fusion/rewrites/MMR",
        len(docs_for_answer),
    )

    return RetrievalOutput(documents=docs_for_answer, clarify=clarify_msg)
