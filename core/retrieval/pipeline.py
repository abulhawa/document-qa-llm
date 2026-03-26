from datetime import datetime, timezone
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
    variants_with_weights: Sequence[Tuple[str, float | None]],
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


def _dedup_bm25_by_id_keep_best_score(hits: Sequence[DocHit]) -> List[DocHit]:
    deduped: List[DocHit] = []
    index_by_id: dict[str, int] = {}
    for hit in hits:
        raw_id = hit.get("_id")
        if raw_id is None:
            deduped.append(hit)
            continue

        key = str(raw_id)
        existing_idx = index_by_id.get(key)
        if existing_idx is None:
            index_by_id[key] = len(deduped)
            deduped.append(hit)
            continue

        if hit.get("score", 0.0) > deduped[existing_idx].get("score", 0.0):
            deduped[existing_idx] = hit

    return deduped


def _to_clamped_float(value: object, *, min_value: float, max_value: float) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if numeric < min_value:
        return min_value
    if numeric > max_value:
        return max_value
    return numeric


def _apply_authority_boost(docs: Sequence[DocHit], cfg: RetrievalConfig) -> None:
    if not cfg.authority_boost_enabled or cfg.authority_boost_weight <= 0:
        return

    for doc in docs:
        base_score = float(doc.get("retrieval_score", 0.0) or 0.0)
        if base_score <= 0:
            continue

        rank = _to_clamped_float(doc.get("authority_rank"), min_value=0.0, max_value=1.0)
        if rank is None:
            continue

        raw_boost = cfg.authority_boost_weight * rank
        max_boost = base_score * max(cfg.authority_boost_max_fraction, 0.0)
        boost = min(raw_boost, max_boost)
        if boost <= 0:
            continue

        doc["retrieval_score"] = base_score + boost


def _parse_epoch_seconds(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _apply_recency_boost(docs: Sequence[DocHit], cfg: RetrievalConfig) -> None:
    if not cfg.recency_boost_enabled or cfg.recency_boost_weight <= 0:
        return

    half_life_days = max(cfg.recency_boost_half_life_days, 1e-6)
    timestamps = [_parse_epoch_seconds(doc.get("modified_at")) for doc in docs]
    valid_timestamps = [ts for ts in timestamps if ts is not None]
    if not valid_timestamps:
        return

    newest_ts = max(valid_timestamps)
    for doc, ts in zip(docs, timestamps):
        if ts is None:
            continue

        base_score = float(doc.get("retrieval_score", 0.0) or 0.0)
        if base_score <= 0:
            continue

        age_days = max((newest_ts - ts) / 86400.0, 0.0)
        freshness = 0.5 ** (age_days / half_life_days)
        raw_boost = cfg.recency_boost_weight * freshness
        max_boost = base_score * max(cfg.recency_boost_max_fraction, 0.0)
        boost = min(raw_boost, max_boost)
        if boost <= 0:
            continue

        doc["retrieval_score"] = base_score + boost


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
    bm25_results_all = _dedup_bm25_by_id_keep_best_score(bm25_results_all)

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
    _apply_authority_boost(fused, cfg)
    _apply_recency_boost(fused, cfg)

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

    # If short, top-up from remaining non-duplicate kept docs first.
    if len(docs_for_answer) < cfg.top_k and kept:
        selected_doc_ids = {id(doc) for doc in docs_for_answer}
        need = cfg.top_k - len(docs_for_answer)
        remaining_kept = [doc for doc in kept if id(doc) not in selected_doc_ids]
        if remaining_kept:
            docs_for_answer.extend(remaining_kept[:need])

    # Optionally top-up from duplicates only when unique docs are insufficient.
    if (
        cfg.include_dups_if_needed
        and len(kept) < cfg.top_k
        and len(docs_for_answer) < cfg.top_k
        and dups
    ):
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
