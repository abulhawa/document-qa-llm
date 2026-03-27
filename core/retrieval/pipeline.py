import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set, Tuple

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
from core.query_rewriter import has_strong_query_anchors

_CV_FAMILY_DOC_TYPES: Set[str] = {"cv", "resume"}
_PROFILE_DOC_TYPES: Set[str] = {
    "cv",
    "resume",
    "cover_letter",
    "reference_letter",
    "profile",
}
_PROFILE_QUERY_FACT_TERMS: Set[str] = {
    "cv",
    "resume",
    "profile",
    "background",
    "education",
    "study",
    "studies",
    "degree",
    "phd",
    "msc",
    "bsc",
    "experience",
    "university",
    "college",
}
_PROFILE_QUERY_PERSON_TERMS: Set[str] = {
    "who",
    "where",
    "when",
    "did",
    "person",
    "candidate",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_OUT_OF_CORPUS_CUE_TERMS: Set[str] = {
    "bitcoin",
    "crypto",
    "price",
    "weather",
    "forecast",
    "bundesliga",
    "match",
    "score",
    "today",
    "tomorrow",
    "yesterday",
    "stock",
    "news",
}
_LIVE_RECENCY_TERMS: Set[str] = {
    "today",
    "tomorrow",
    "yesterday",
    "current",
    "latest",
    "now",
}
_CORPUS_DOMAIN_ANCHOR_TERMS: Set[str] = {
    "cv",
    "resume",
    "cover",
    "letter",
    "reference",
    "report",
    "paper",
    "course",
    "lecture",
    "project",
    "contract",
    "policy",
    "invoice",
    "payroll",
    "insurance",
    "government",
    "jobcenter",
    "form",
    "tax",
    "receipt",
    "academic",
}
_QUERY_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "what",
    "who",
    "where",
    "when",
    "why",
    "how",
    "in",
    "on",
    "for",
    "of",
    "to",
    "and",
    "or",
}
_CANONICAL_DOC_QUERY_TERMS: Set[str] = {
    "cv",
    "resume",
    "cover",
    "letter",
    "reference",
    "report",
    "paper",
    "course",
    "lecture",
    "contract",
    "policy",
    "invoice",
    "receipt",
    "jobcenter",
    "form",
    "formular",
    "insurance",
    "document",
    "doc",
    "file",
    "pdf",
}
_AMBIGUOUS_PRONOUNS: Set[str] = {"he", "she", "they", "it", "him", "her", "them"}
_GENERIC_LIST_INDEX_TERMS: Set[str] = {
    "list",
    "index",
    "inventory",
    "catalog",
    "manifest",
    "directory",
    "register",
    "toc",
}
_GENERIC_AGGREGATE_SUMMARY_TERMS: Set[str] = {
    "summary",
    "overview",
    "profile",
    "profiles",
    "aggregate",
    "digest",
}
_GENERIC_HARD_NEGATIVE_TITLE_TERMS: Set[str] = (
    _GENERIC_LIST_INDEX_TERMS | _GENERIC_AGGREGATE_SUMMARY_TERMS
)


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


def _tokenize_lower(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _query_content_tokens(query: str) -> Set[str]:
    return {
        tok
        for tok in _tokenize_lower(query)
        if (len(tok) >= 3 or tok == "cv") and tok not in _QUERY_STOPWORDS
    }


def _title_or_filename(doc: DocHit) -> str:
    filename = str(doc.get("filename") or "").strip()
    if filename:
        return filename
    path = str(doc.get("path") or "").strip()
    if not path:
        return ""
    parts = re.split(r"[\\/]+", path)
    return parts[-1] if parts else path


def _title_filename_overlap(
    query_tokens: Set[str],
    doc: DocHit,
) -> tuple[int, float]:
    if not query_tokens:
        return 0, 0.0
    title_tokens = _tokenize_lower(_title_or_filename(doc))
    if not title_tokens:
        return 0, 0.0
    overlap_count = len(query_tokens & title_tokens)
    denominator = min(len(query_tokens), len(title_tokens))
    if denominator <= 0:
        return overlap_count, 0.0
    overlap_ratio = overlap_count / float(denominator)
    return overlap_count, overlap_ratio


def _generic_hard_negative_title_terms(doc: DocHit) -> Set[str]:
    title = _title_or_filename(doc)
    if not title:
        return set()
    title_tokens = _tokenize_lower(title)
    return title_tokens & _GENERIC_HARD_NEGATIVE_TITLE_TERMS


def _is_canonical_anchored_or_semi_anchored_query(
    query: str,
    *,
    anchored_query: bool,
) -> bool:
    tokens = _tokenize_lower(query)
    if not tokens:
        return False
    has_canonical_doc_anchor = bool(tokens & _CANONICAL_DOC_QUERY_TERMS)
    if not has_canonical_doc_anchor:
        return False
    if anchored_query:
        return True
    return bool(tokens & _AMBIGUOUS_PRONOUNS)


def _apply_canonical_lexical_rescue(
    query: str,
    docs: Sequence[DocHit],
    cfg: RetrievalConfig,
    *,
    anchored_query: bool,
) -> None:
    if not cfg.canonical_lexical_rescue_enabled or len(docs) < 2:
        return
    if not _is_canonical_anchored_or_semi_anchored_query(
        query,
        anchored_query=anchored_query,
    ):
        return

    query_tokens = _query_content_tokens(query)
    if not query_tokens:
        return

    winner_idx = max(
        range(len(docs)),
        key=lambda idx: (
            float(docs[idx].get("retrieval_score", 0.0) or 0.0),
            str(docs[idx].get("modified_at", "")),
        ),
    )
    winner = docs[winner_idx]
    winner_score = float(winner.get("retrieval_score", 0.0) or 0.0)
    winner_vector = float(winner.get("score_vector", 0.0) or 0.0)
    winner_bm25 = float(winner.get("score_bm25", 0.0) or 0.0)
    winner_overlap_count, winner_overlap_ratio = _title_filename_overlap(
        query_tokens, winner
    )

    overlap_by_idx: Dict[int, tuple[int, float]] = {
        idx: _title_filename_overlap(query_tokens, doc)
        for idx, doc in enumerate(docs)
    }
    lexical_idx = max(
        range(len(docs)),
        key=lambda idx: (
            float(docs[idx].get("score_bm25", 0.0) or 0.0),
            overlap_by_idx[idx][1],
            overlap_by_idx[idx][0],
            float(docs[idx].get("retrieval_score", 0.0) or 0.0),
        ),
    )
    if lexical_idx == winner_idx:
        return

    candidate = docs[lexical_idx]
    candidate_score = float(candidate.get("retrieval_score", 0.0) or 0.0)
    if winner_score <= candidate_score:
        return

    score_gap = winner_score - candidate_score
    if score_gap > max(cfg.canonical_lexical_rescue_max_score_gap, 0.0):
        return

    candidate_vector = float(candidate.get("score_vector", 0.0) or 0.0)
    if winner_vector <= candidate_vector:
        return

    candidate_bm25 = float(candidate.get("score_bm25", 0.0) or 0.0)
    bm25_advantage = candidate_bm25 - winner_bm25
    if candidate_bm25 < max(cfg.canonical_lexical_rescue_min_bm25, 0.0):
        return
    if bm25_advantage < max(cfg.canonical_lexical_rescue_min_bm25_advantage, 0.0):
        return

    candidate_overlap_count, candidate_overlap_ratio = overlap_by_idx[lexical_idx]
    overlap_floor_count = max(int(cfg.canonical_lexical_rescue_min_title_overlap_count), 0)
    overlap_floor_ratio = max(float(cfg.canonical_lexical_rescue_min_title_overlap_ratio), 0.0)
    candidate_overlap_is_strong = (
        candidate_overlap_count >= overlap_floor_count
        or candidate_overlap_ratio >= overlap_floor_ratio
    )
    title_filename_edge = candidate_overlap_is_strong and (
        candidate_overlap_count > winner_overlap_count
        or candidate_overlap_ratio > winner_overlap_ratio
    )
    if (
        not title_filename_edge
        and bm25_advantage
        < max(cfg.canonical_lexical_rescue_strong_bm25_advantage, 0.0)
    ):
        return

    candidate["retrieval_score"] = winner_score + 1e-6
    candidate["_canonical_lexical_rescue"] = {
        "winner_checksum": winner.get("checksum"),
        "score_gap_closed": round(score_gap, 6),
        "winner_vector_advantage": round(winner_vector - candidate_vector, 6),
        "bm25_advantage": round(bm25_advantage, 6),
        "candidate_title_overlap_count": candidate_overlap_count,
        "candidate_title_overlap_ratio": round(candidate_overlap_ratio, 4),
        "winner_title_overlap_count": winner_overlap_count,
        "winner_title_overlap_ratio": round(winner_overlap_ratio, 4),
    }


def _apply_canonical_hard_negative_suppression(
    query: str,
    docs: Sequence[DocHit],
    cfg: RetrievalConfig,
    *,
    anchored_query: bool,
) -> None:
    if not cfg.canonical_hard_negative_suppression_enabled or len(docs) < 2:
        return
    if not _is_canonical_anchored_or_semi_anchored_query(
        query,
        anchored_query=anchored_query,
    ):
        return

    query_tokens = _query_content_tokens(query)
    if not query_tokens:
        return

    winner_idx = max(
        range(len(docs)),
        key=lambda idx: (
            float(docs[idx].get("retrieval_score", 0.0) or 0.0),
            str(docs[idx].get("modified_at", "")),
        ),
    )
    winner = docs[winner_idx]
    winner_score = float(winner.get("retrieval_score", 0.0) or 0.0)
    winner_vector = float(winner.get("score_vector", 0.0) or 0.0)
    winner_overlap_count, winner_overlap_ratio = _title_filename_overlap(
        query_tokens, winner
    )
    generic_terms = _generic_hard_negative_title_terms(winner)
    if not generic_terms:
        return
    if winner_vector < max(cfg.canonical_hard_negative_min_winner_vector_score, 0.0):
        return
    if (
        winner_overlap_count
        > max(int(cfg.canonical_hard_negative_max_winner_title_overlap_count), 0)
    ):
        return

    min_candidate_overlap_count = max(
        int(cfg.canonical_hard_negative_min_candidate_title_overlap_count), 1
    )
    max_score_gap = max(cfg.canonical_hard_negative_max_score_gap, 0.0)
    best_candidate_idx: int | None = None
    best_candidate_overlap_count = -1
    best_candidate_score = float("-inf")
    for idx, candidate in enumerate(docs):
        if idx == winner_idx:
            continue
        if _generic_hard_negative_title_terms(candidate):
            continue
        candidate_overlap_count, _ = _title_filename_overlap(query_tokens, candidate)
        if candidate_overlap_count < min_candidate_overlap_count:
            continue
        candidate_score = float(candidate.get("retrieval_score", 0.0) or 0.0)
        if (winner_score - candidate_score) > max_score_gap:
            continue
        if (
            best_candidate_idx is None
            or candidate_overlap_count > best_candidate_overlap_count
            or (
                candidate_overlap_count == best_candidate_overlap_count
                and candidate_score > best_candidate_score
            )
        ):
            best_candidate_idx = idx
            best_candidate_overlap_count = candidate_overlap_count
            best_candidate_score = candidate_score

    if best_candidate_idx is None:
        return

    candidate = docs[best_candidate_idx]
    candidate["retrieval_score"] = winner_score + 1e-6
    candidate["_canonical_hard_negative_suppression"] = {
        "suppressed_winner_checksum": winner.get("checksum"),
        "suppressed_winner_generic_terms": sorted(generic_terms),
        "winner_score_gap_closed": round(winner_score - best_candidate_score, 6),
        "winner_title_overlap_count": winner_overlap_count,
        "winner_title_overlap_ratio": round(winner_overlap_ratio, 4),
        "candidate_title_overlap_count": best_candidate_overlap_count,
    }


def _is_profile_intent_query(query: str) -> bool:
    tokens = _tokenize_lower(query)
    if not tokens:
        return False
    has_fact_signal = bool(tokens & _PROFILE_QUERY_FACT_TERMS)
    has_person_signal = bool(tokens & _PROFILE_QUERY_PERSON_TERMS)
    return has_fact_signal and has_person_signal


def _is_profile_class_doc(doc: DocHit) -> bool:
    doc_type = str(doc.get("doc_type") or "").strip().lower()
    if doc_type in _PROFILE_DOC_TYPES:
        return True
    person_name = str(doc.get("person_name") or "").strip()
    return bool(person_name)


def _apply_profile_intent_adjustment(query: str, docs: Sequence[DocHit], cfg: RetrievalConfig) -> None:
    if not cfg.profile_intent_boost_enabled or cfg.profile_intent_boost_weight <= 0:
        return
    if not _is_profile_intent_query(query):
        return

    max_fraction = max(cfg.profile_intent_boost_max_fraction, 0.0)
    for doc in docs:
        base_score = float(doc.get("retrieval_score", 0.0) or 0.0)
        if base_score <= 0:
            continue

        doc["_profile_intent_match"] = True
        if _is_profile_class_doc(doc):
            raw_adjustment = cfg.profile_intent_boost_weight
            max_adjustment = base_score * max_fraction
            adjustment = min(raw_adjustment, max_adjustment)
            if adjustment <= 0:
                continue
            doc["retrieval_score"] = base_score + adjustment
            doc["_profile_intent_adjustment"] = adjustment
            doc["_profile_intent_class"] = "profile"
            continue

        raw_adjustment = cfg.profile_intent_boost_weight * 0.5
        max_adjustment = base_score * max_fraction * 0.5
        adjustment = min(raw_adjustment, max_adjustment)
        if adjustment <= 0:
            continue
        doc["retrieval_score"] = max(base_score - adjustment, 0.0)
        doc["_profile_intent_adjustment"] = -adjustment
        doc["_profile_intent_class"] = "other"


def _cv_family_key(doc: DocHit) -> str | None:
    doc_type = str(doc.get("doc_type") or "").strip().lower()
    if doc_type not in _CV_FAMILY_DOC_TYPES:
        return None
    person_name = str(doc.get("person_name") or "").strip().lower()
    person_name = re.sub(r"\s+", " ", person_name)
    if not person_name:
        return None
    return person_name


def _collapse_cv_families(docs: Sequence[DocHit], cfg: RetrievalConfig) -> List[DocHit]:
    if not cfg.cv_family_collapse_enabled:
        return list(docs)

    grouped: Dict[str, List[DocHit]] = {}
    for doc in docs:
        key = _cv_family_key(doc)
        if key is None:
            continue
        grouped.setdefault(key, []).append(doc)

    if not grouped:
        return list(docs)

    selected_by_family: Dict[str, DocHit] = {}
    margin = max(cfg.cv_family_relevance_margin, 0.0)
    for family_key, members in grouped.items():
        scored_members: List[Tuple[DocHit, float, float | None]] = []
        for member in members:
            scored_members.append(
                (
                    member,
                    float(member.get("retrieval_score", 0.0) or 0.0),
                    _parse_epoch_seconds(member.get("modified_at")),
                )
            )

        best_doc, best_score, _ = max(scored_members, key=lambda item: (item[1], item[2] or float("-inf")))

        reason = "highest_relevance"
        selected = best_doc
        with_timestamps = [item for item in scored_members if item[2] is not None]
        if with_timestamps:
            newest_doc, newest_score, _ = max(with_timestamps, key=lambda item: (item[2], item[1]))
            if newest_doc is not best_doc:
                if best_score - newest_score <= margin:
                    selected = newest_doc
                    reason = "newest_within_margin"
                else:
                    reason = "older_higher_relevance"

        selected["_cv_family_key"] = family_key
        selected["_cv_family_size"] = len(members)
        selected["_cv_family_suppressed"] = len(members) - 1
        selected["_cv_family_choice_reason"] = reason
        selected_by_family[family_key] = selected

    collapsed: List[DocHit] = []
    emitted_families: Set[str] = set()
    for doc in docs:
        family_key = _cv_family_key(doc)
        if family_key is None:
            collapsed.append(doc)
            continue
        if family_key in emitted_families:
            continue
        collapsed.append(selected_by_family[family_key])
        emitted_families.add(family_key)

    return sorted(
        collapsed,
        key=lambda item: (item.get("retrieval_score", 0.0), item.get("modified_at", "")),
        reverse=True,
    )


def _should_abstain_for_out_of_corpus_query(
    query: str,
    docs: Sequence[DocHit],
    cfg: RetrievalConfig,
) -> bool:
    if not cfg.abstention_enabled or not docs:
        return False

    query_terms = _query_content_tokens(query)
    if not query_terms:
        return False

    if not (query_terms & _OUT_OF_CORPUS_CUE_TERMS):
        return False
    if query_terms & _CORPUS_DOMAIN_ANCHOR_TERMS:
        return False
    if query_terms & _LIVE_RECENCY_TERMS:
        return True

    max_overlap_terms = 0
    top_docs = docs[: min(len(docs), 3)]
    for doc in top_docs:
        doc_text = " ".join(
            str(doc.get(field) or "")
            for field in ("text", "path", "filename", "doc_type", "person_name")
        )
        overlap_terms = len(query_terms & _tokenize_lower(doc_text))
        if overlap_terms > max_overlap_terms:
            max_overlap_terms = overlap_terms

    return max_overlap_terms < max(cfg.abstention_min_overlap_terms, 1)


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
    exact_query = query.strip()
    anchored_query = has_strong_query_anchors(exact_query)
    use_exact_only = (
        cfg.enable_variants
        and cfg.anchored_exact_only
        and anchored_query
    )
    if use_exact_only:
        logger.info(
            "Skipping rewrite variants for strongly anchored query; using exact query only."
        )
        variants_with_weights = [(exact_query, cfg.weight_exact_bm25)]
    elif cfg.enable_variants:
        variants_result = generate_variants(query)
        if "clarify" in variants_result:
            clarify_msg = variants_result["clarify"]
            return RetrievalOutput(documents=[], clarify=clarify_msg)
        variants_with_weights = variants_result.get("variants", [])
    if not variants_with_weights:
        variants_with_weights = [(exact_query, cfg.weight_exact_bm25)]

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
    fusion_weight_vector = cfg.fusion_weight_vector
    fusion_weight_bm25 = cfg.fusion_weight_bm25
    if anchored_query and cfg.anchored_lexical_bias_enabled:
        fusion_weight_vector = cfg.anchored_fusion_weight_vector
        fusion_weight_bm25 = cfg.anchored_fusion_weight_bm25

    fused = fuse_semantic_and_bm25(
        vector_results_all,
        bm25_results_all,
        w_vec=fusion_weight_vector,
        w_bm25=fusion_weight_bm25,
    )
    for d in fused:
        w = d.get("_bm25_variant_weight", 1.0)
        d["retrieval_score"] = fusion_weight_vector * d.get("score_vector", 0.0) + fusion_weight_bm25 * (
            d.get("score_bm25", 0.0) * w
        )
    _apply_authority_boost(fused, cfg)
    _apply_recency_boost(fused, cfg)
    _apply_profile_intent_adjustment(query, fused, cfg)
    _apply_canonical_lexical_rescue(
        exact_query,
        fused,
        cfg,
        anchored_query=anchored_query,
    )
    _apply_canonical_hard_negative_suppression(
        exact_query,
        fused,
        cfg,
        anchored_query=anchored_query,
    )

    fused_sorted: List[DocHit] = sorted(
        fused,
        key=lambda x: (x.get("retrieval_score", 0.0), x.get("modified_at", "")),
        reverse=True,
    )
    unique_docs = dedup_by_checksum(fused_sorted)
    unique_docs = _collapse_cv_families(unique_docs, cfg)

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

    if _should_abstain_for_out_of_corpus_query(query, docs_for_answer, cfg):
        logger.info(
            "Abstention gate triggered: suppressing low-overlap results for out-of-corpus style query."
        )
        return RetrievalOutput(documents=[], clarify=None)

    logger.info(
        "Retrieval returned %s results after fusion/rewrites/MMR",
        len(docs_for_answer),
    )

    return RetrievalOutput(documents=docs_for_answer, clarify=clarify_msg)
