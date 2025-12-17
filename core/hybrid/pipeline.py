from typing import List, Dict, Any, Optional, Tuple

from config import logger

from core.vector_store import retrieve_top_k as semantic_retriever
from core.opensearch_store import search as keyword_retriever

from core.hybrid.fusion import fuse_semantic_and_bm25, dedup_by_checksum
from core.hybrid.variants import generate_variants
from core.hybrid.mmr import mmr_select
from core.hybrid.dedup import collapse_near_duplicates
from core.hybrid.reranker import CrossEncoderReranker


def retrieve_hybrid(
    query: str,
    top_k_each: int = 20,
    final_k: int = 5,
    embed_texts=None,
    cross_encoder: CrossEncoderReranker | None = None,
    mmr_k: int = 8,
    mmr_lambda: float = 0.6,
    sim_threshold: float = 0.90,
    include_dups_if_needed: bool = True,
    variants: Optional[List[Tuple[str, float]]] = None,
    weight_exact_bm25: float = 1.0,
    weight_rewrite_bm25: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining BM25 (OpenSearch) and vector (Qdrant) results.
    Deduplicates by checksum; preserves existing behavior and logging.
    """
    logger.info(f"Running hybrid search: '{query}'")

    # 1) Generate variants (keep your rewritten as #1)
    # If caller provides variants, use them; otherwise generate.
    if variants is None:
        variants_result = generate_variants(query)
        if "clarify" in variants_result:
            return [{"clarify": variants_result["clarify"]}]
        variants_with_weights = variants_result.get("variants", [])
    else:
        variants_with_weights = variants

    if not variants_with_weights:
        variants_with_weights = [(query, weight_exact_bm25)]

    # 2) Retrieve per variant and collect with weights
    vector_results_all: List[Dict[str, Any]] = []
    bm25_results_all: List[Dict[str, Any]] = []

    for idx, (qv, weight_override) in enumerate(variants_with_weights):
        v = semantic_retriever(qv, top_k=top_k_each)  # Qdrant
        vector_results_all.extend(v)

        b = keyword_retriever(qv, top_k=top_k_each)  # OpenSearch
        # attach a lightweight weight signal used later in fusion
        w_default = weight_exact_bm25 if idx == 0 else weight_rewrite_bm25
        w_effective = weight_override if weight_override is not None else w_default
        for hit in b:
            hit["_bm25_variant_weight"] = w_effective
        bm25_results_all.extend(b)

    # 3) Fuse with normalization (respect variant weights)
    fused = fuse_semantic_and_bm25(
        vector_results_all, bm25_results_all, w_vec=0.7, w_bm25=0.3
    )
    for d in fused:
        w = d.get("_bm25_variant_weight", 1.0)
        d["hybrid_score"] = 0.7 * d.get("score_vector", 0.0) + 0.3 * (
            d.get("score_bm25", 0.0) * w
        )

    fused_sorted = sorted(
        fused,
        key=lambda x: (x["hybrid_score"], x.get("modified_at", "")),
        reverse=True,
    )
    unique_docs = dedup_by_checksum(fused_sorted)

    # 4) Near-duplicate collapse (keep one rep; stage dups aside)
    kept, dups = (unique_docs, [])
    if embed_texts is not None:
        kept, dups = collapse_near_duplicates(
            unique_docs,
            embed_texts=embed_texts,
            sim_threshold=sim_threshold,
            keep_limit=64,
        )

    # 5) MMR (optional) for diversity
    if embed_texts is not None:
        docs_for_answer = mmr_select(
            query,
            kept,
            embed=embed_texts,
            k=min(mmr_k, final_k),
            lambda_mult=mmr_lambda,
        )
    else:
        docs_for_answer = kept[:final_k]

    # If short, optionally top-up from duplicates
    if include_dups_if_needed and len(docs_for_answer) < final_k and dups:
        need = final_k - len(docs_for_answer)
        docs_for_answer.extend(dups[:need])

    # Optional cross-encoder rerank
    if cross_encoder is not None and docs_for_answer:
        docs_for_answer = cross_encoder.rerank(
            query, docs_for_answer, top_n=len(docs_for_answer)
        )

    logger.info(
        f"Hybrid search returned {len(docs_for_answer)} results after fusion/rewrites/MMR"
    )

    return docs_for_answer
