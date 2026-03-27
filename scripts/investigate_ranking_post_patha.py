#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Ensure repo root is on sys.path when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FULLTEXT_INDEX
from core.opensearch_client import get_client
from core.query_rewriter import has_strong_query_anchors
from core.retrieval.pipeline import retrieve
from core.retrieval.types import RetrievalConfig


TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")

RANK_BUCKET_1 = "rank_1"
RANK_BUCKET_2_3 = "rank_2_to_3"
RANK_BUCKET_4_10 = "rank_4_to_10"
RANK_BUCKET_11_TO_DEPTH = "rank_11_to_depth"
RANK_BUCKET_NOT_RETRIEVED = "not_retrieved_within_depth"

LIKELY_CORRECT = "likely_correct_despite_rank_miss"
POSSIBLY_CORRECT_TOP3 = "possibly_correct_with_top3_context"
POSSIBLY_CORRECT_SIMILAR = "possibly_correct_due_to_high_similarity"
UNLIKELY_OR_UNKNOWN = "unlikely_or_unknown"

DEFAULT_PROBE_DEPTH = 60
DEFAULT_REVIEW_RANK_LIMIT = 5
EQUIV_SUPPORT_CONTAINMENT_MIN = 0.82
EQUIV_SUPPORT_SEQUENCE_MIN = 0.72
EQUIV_SUPPORT_ALT_CONTAINMENT_MIN = 0.75
EQUIV_SUPPORT_ALT_JACCARD_MIN = 0.50
EQUIV_SUPPORT_ALT_SEQUENCE_MIN = 0.65
REVIEW_QUERY_CONTAINMENT_MIN = 0.68
REVIEW_QUERY_SEQUENCE_MIN = 0.58
REVIEW_QUERY_JACCARD_MIN = 0.30
REVIEW_ANCHOR_COVERAGE_MIN = 0.50
DEFAULT_SUPPORT_LABELS_PATH = Path("tests/fixtures/retrieval_eval_answer_support_labels.json")
BENCHMARK_MODE_STRICT_RETRIEVAL = "strict_retrieval"
BENCHMARK_MODE_ANSWER_SUPPORT = "answer_support"
QUERY_TYPE_CANONICAL_DOCUMENT = "canonical_document_query"
QUERY_TYPE_MULTI_SOURCE_FACTUAL = "multi_source_factual_query"
QUERY_TYPE_AMBIGUOUS_REVIEWER_NEEDED = "ambiguous_reviewer_needed"
QUERY_TYPE_ORDER: tuple[str, ...] = (
    QUERY_TYPE_CANONICAL_DOCUMENT,
    QUERY_TYPE_MULTI_SOURCE_FACTUAL,
    QUERY_TYPE_AMBIGUOUS_REVIEWER_NEEDED,
)
VALID_QUERY_TYPES = {
    QUERY_TYPE_CANONICAL_DOCUMENT,
    QUERY_TYPE_MULTI_SOURCE_FACTUAL,
    QUERY_TYPE_AMBIGUOUS_REVIEWER_NEEDED,
}
DEFAULT_POSITIVE_QUERY_TYPE = QUERY_TYPE_CANONICAL_DOCUMENT
RANKING_INVESTIGATION_SCHEMA_VERSION = "ranking_investigation.v5"
RANKING_CAUSE_VECTOR_DOMINANCE = "vector dominance"
RANKING_CAUSE_TITLE_FILENAME_UNDERWEIGHTING = "title/filename underweighting"
RANKING_CAUSE_SIBLING_COLLISION = "sibling/near-duplicate collision"
RANKING_CAUSE_CHUNK_AGGREGATION_BIAS = "chunk aggregation bias"
RANKING_CAUSE_DOC_TYPE_PRIOR_SUPPRESSION = "doc-type prior suppression"
RANKING_CAUSE_CANDIDATE_GENERATION_MISS = "candidate generation miss"
RANKING_CAUSE_AMBIGUOUS = "ambiguous/manual review"
RANKING_CAUSE_ORDER: tuple[str, ...] = (
    RANKING_CAUSE_VECTOR_DOMINANCE,
    RANKING_CAUSE_TITLE_FILENAME_UNDERWEIGHTING,
    RANKING_CAUSE_SIBLING_COLLISION,
    RANKING_CAUSE_CHUNK_AGGREGATION_BIAS,
    RANKING_CAUSE_DOC_TYPE_PRIOR_SUPPRESSION,
    RANKING_CAUSE_CANDIDATE_GENERATION_MISS,
    RANKING_CAUSE_AMBIGUOUS,
)
COMMON_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
}
GENERIC_LIST_INDEX_TERMS = {
    "list",
    "index",
    "inventory",
    "catalog",
    "manifest",
    "directory",
    "register",
    "toc",
}
GENERIC_AGGREGATE_SUMMARY_TERMS = {
    "summary",
    "overview",
    "profile",
    "profiles",
    "aggregate",
    "digest",
}
GENERIC_HARD_NEGATIVE_TERMS = GENERIC_LIST_INDEX_TERMS | GENERIC_AGGREGATE_SUMMARY_TERMS
HARD_NEGATIVE_CLASS_GENERIC_LIST_INDEX = "generic_list_index_file"
HARD_NEGATIVE_CLASS_AGGREGATE_SUMMARY = "aggregate_or_profile_summary_file"
HARD_NEGATIVE_CLASS_SIBLING_FAMILY_COLLISION = "sibling_family_collision"
HARD_NEGATIVE_CLASS_WEAK_TITLE_VECTOR = "weak_title_alignment_vector_driven"
HARD_NEGATIVE_CLASS_ORDER: tuple[str, ...] = (
    HARD_NEGATIVE_CLASS_GENERIC_LIST_INDEX,
    HARD_NEGATIVE_CLASS_AGGREGATE_SUMMARY,
    HARD_NEGATIVE_CLASS_SIBLING_FAMILY_COLLISION,
    HARD_NEGATIVE_CLASS_WEAK_TITLE_VECTOR,
)
LIKELY_RANKING_FIX_CANDIDATE = "likely_ranking_fix_candidate"
LIKELY_BENCHMARK_AMBIGUITY = "likely_benchmark_ambiguity_manual_review"
DEFAULT_ARTIFACT_NEAR_TIE_EPSILON = 1e-5
STRICT_CANONICAL_CLEANED_RESIDUAL_SCHEMA_VERSION = (
    "strict_canonical_cleaned_residuals.v1"
)


def reciprocal_rank(rank: int | None) -> float:
    if rank is None or rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def rank_bucket(rank: int | None, depth: int) -> str:
    if rank is None or rank > depth:
        return RANK_BUCKET_NOT_RETRIEVED
    if rank == 1:
        return RANK_BUCKET_1
    if rank <= 3:
        return RANK_BUCKET_2_3
    if rank <= 10:
        return RANK_BUCKET_4_10
    return RANK_BUCKET_11_TO_DEPTH


def text_similarity_metrics(text_a: str, text_b: str) -> dict[str, float | bool]:
    sample_a = _sample_text(text_a)
    sample_b = _sample_text(text_b)
    tokens_a = set(TOKEN_RE.findall(sample_a.lower()))
    tokens_b = set(TOKEN_RE.findall(sample_b.lower()))

    jaccard = 0.0
    containment = 0.0
    if tokens_a and tokens_b:
        inter = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jaccard = inter / union if union > 0 else 0.0
        containment = inter / min(len(tokens_a), len(tokens_b))

    seq_ratio = SequenceMatcher(None, sample_a, sample_b).ratio() if (sample_a and sample_b) else 0.0
    near_duplicate = (
        containment >= 0.95
        or seq_ratio >= 0.95
        or (containment >= 0.90 and jaccard >= 0.85 and seq_ratio >= 0.85)
    )
    return {
        "jaccard": round(jaccard, 4),
        "containment_min": round(containment, 4),
        "sequence_ratio": round(seq_ratio, 4),
        "near_duplicate": near_duplicate,
    }


def auto_answer_likelihood(
    *,
    hit_at_3: bool,
    expected_rank: int | None,
    near_duplicate: bool,
    similarity_containment: float,
) -> str:
    if near_duplicate:
        return LIKELY_CORRECT
    if hit_at_3 and expected_rank is not None and expected_rank <= 3:
        return POSSIBLY_CORRECT_TOP3
    if expected_rank is not None and expected_rank <= 10 and similarity_containment >= 0.80:
        return POSSIBLY_CORRECT_SIMILAR
    return UNLIKELY_OR_UNKNOWN


def query_anchor_tokens(query: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall((query or "").lower()):
        if len(token) < 3 or token in COMMON_STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _anchor_conditioned_slice(
    text: str,
    anchors: Sequence[str],
    *,
    max_segments: int = 80,
) -> tuple[str, float]:
    if not text:
        return "", 0.0
    if not anchors:
        return _sample_text(text, max_chars=8000), 0.0

    selected: list[str] = []
    covered: set[str] = set()
    anchor_set = set(anchors)
    for segment in SENTENCE_SPLIT_RE.split(text):
        snippet = segment.strip()
        if not snippet:
            continue
        tokens = set(TOKEN_RE.findall(snippet.lower()))
        hits = tokens & anchor_set
        if not hits:
            continue
        covered.update(hits)
        selected.append(snippet)
        if len(selected) >= max_segments:
            break
    if not selected:
        return "", 0.0
    coverage = len(covered) / float(len(anchor_set))
    return _sample_text(" ".join(selected), max_chars=8000), round(coverage, 4)


def query_conditioned_similarity_metrics(
    *,
    query: str,
    candidate_text: str,
    expected_text: str,
) -> dict[str, Any] | None:
    anchors = query_anchor_tokens(query)
    candidate_slice, candidate_anchor_coverage = _anchor_conditioned_slice(
        candidate_text, anchors
    )
    expected_slice, expected_anchor_coverage = _anchor_conditioned_slice(
        expected_text, anchors
    )
    if not candidate_slice or not expected_slice:
        return None
    metrics = text_similarity_metrics(candidate_slice, expected_slice)
    metrics["candidate_anchor_coverage"] = round(candidate_anchor_coverage, 4)
    metrics["expected_anchor_coverage"] = round(expected_anchor_coverage, 4)
    metrics["anchors"] = anchors
    return metrics


def is_equivalent_answer_support(similarity: Mapping[str, Any] | None) -> bool:
    if not similarity:
        return False
    if bool(similarity.get("near_duplicate")):
        return True
    containment = float(similarity.get("containment_min", 0.0) or 0.0)
    sequence_ratio = float(similarity.get("sequence_ratio", 0.0) or 0.0)
    jaccard = float(similarity.get("jaccard", 0.0) or 0.0)
    if containment >= EQUIV_SUPPORT_CONTAINMENT_MIN and sequence_ratio >= EQUIV_SUPPORT_SEQUENCE_MIN:
        return True
    if (
        containment >= EQUIV_SUPPORT_ALT_CONTAINMENT_MIN
        and jaccard >= EQUIV_SUPPORT_ALT_JACCARD_MIN
        and sequence_ratio >= EQUIV_SUPPORT_ALT_SEQUENCE_MIN
    ):
        return True
    return False


def is_review_candidate_similarity(similarity: Mapping[str, Any] | None) -> bool:
    if not similarity:
        return False
    if is_equivalent_answer_support(similarity):
        return True
    containment = float(similarity.get("containment_min", 0.0) or 0.0)
    sequence_ratio = float(similarity.get("sequence_ratio", 0.0) or 0.0)
    jaccard = float(similarity.get("jaccard", 0.0) or 0.0)
    candidate_anchor_coverage = float(similarity.get("candidate_anchor_coverage", 0.0) or 0.0)
    expected_anchor_coverage = float(similarity.get("expected_anchor_coverage", 0.0) or 0.0)
    return (
        containment >= REVIEW_QUERY_CONTAINMENT_MIN
        and sequence_ratio >= REVIEW_QUERY_SEQUENCE_MIN
        and jaccard >= REVIEW_QUERY_JACCARD_MIN
        and candidate_anchor_coverage >= REVIEW_ANCHOR_COVERAGE_MIN
        and expected_anchor_coverage >= REVIEW_ANCHOR_COVERAGE_MIN
    )


def _sample_text(text: str, max_chars: int = 20000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _dedupe_str_checksums(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        checksum = value.strip()
        if not checksum or checksum in seen:
            continue
        seen.add(checksum)
        out.append(checksum)
    return out


def _normalize_query_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in VALID_QUERY_TYPES:
        return normalized
    return None


def _load_support_label_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    data = _load_json(path)
    raw_overrides = data.get("overrides", {})
    if not isinstance(raw_overrides, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for query_id, override in raw_overrides.items():
        if not isinstance(query_id, str) or not isinstance(override, dict):
            continue
        out[query_id] = override
    return out


def _load_support_labels(path: Path | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if path is None or not path.exists():
        return {}, {}
    data = _load_json(path)
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    raw_overrides = data.get("overrides", {})
    if not isinstance(raw_overrides, dict):
        return meta, {}
    overrides: dict[str, dict[str, Any]] = {}
    for query_id, override in raw_overrides.items():
        if not isinstance(query_id, str) or not isinstance(override, dict):
            continue
        overrides[query_id] = override
    return meta, overrides


def resolve_query_benchmark_type(
    *,
    fixture_query: Mapping[str, Any] | None,
    label_override: Mapping[str, Any] | None,
    default_query_type: str = DEFAULT_POSITIVE_QUERY_TYPE,
) -> str:
    override_type = _normalize_query_type(
        (label_override or {}).get("benchmark_query_type")
    )
    if override_type:
        return override_type
    fixture_type = _normalize_query_type((fixture_query or {}).get("benchmark_query_type"))
    if fixture_type:
        return fixture_type
    return _normalize_query_type(default_query_type) or QUERY_TYPE_CANONICAL_DOCUMENT


def benchmark_mode_for_query_type(query_type: str) -> str:
    if query_type == QUERY_TYPE_MULTI_SOURCE_FACTUAL:
        return BENCHMARK_MODE_ANSWER_SUPPORT
    return BENCHMARK_MODE_STRICT_RETRIEVAL


def resolve_support_expectations(
    *,
    strict_expected_checksums: Sequence[str],
    label_override: Mapping[str, Any] | None,
) -> tuple[list[str], str | None]:
    strict = _dedupe_str_checksums(strict_expected_checksums)
    if not label_override:
        preferred = strict[0] if strict else None
        return strict, preferred

    raw_mode = str(label_override.get("answer_support_mode") or "merge").strip().lower()
    mode = raw_mode if raw_mode in {"merge", "replace"} else "merge"
    extra_checksums = _dedupe_str_checksums(
        label_override.get("answer_support_checksums") or []
    )
    if mode == "replace":
        support = extra_checksums
    else:
        support = _dedupe_str_checksums([*strict, *extra_checksums])

    preferred_override = label_override.get("preferred_checksum")
    preferred_checksum: str | None = None
    if isinstance(preferred_override, str) and preferred_override.strip():
        preferred_checksum = preferred_override.strip()
    if preferred_checksum is None:
        preferred_checksum = strict[0] if strict else None
    return support, preferred_checksum


def _fetch_fulltext_by_checksum(
    os_client: Any,
    checksum: str,
) -> dict[str, Any] | None:
    try:
        response = os_client.get(index=FULLTEXT_INDEX, id=checksum)
        source = response.get("_source", {}) or {}
        source["id"] = response.get("_id") or checksum
        return source
    except Exception:
        pass

    try:
        response = os_client.search(
            index=FULLTEXT_INDEX,
            body={
                "size": 1,
                "query": {"term": {"checksum": {"value": checksum}}},
            },
        )
    except Exception:
        return None
    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return None
    hit = hits[0]
    source = hit.get("_source", {}) or {}
    source["id"] = hit.get("_id") or checksum
    return source


def _build_probe_cfg(patha_cfg: Mapping[str, Any], probe_depth: int) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=max(probe_depth, 1),
        top_k_each=max(probe_depth * 4, 40),
        enable_variants=False,
        enable_mmr=bool(patha_cfg.get("enable_mmr", True)),
        anchored_exact_only=bool(patha_cfg.get("anchored_exact_only", True)),
        anchored_lexical_bias_enabled=bool(
            patha_cfg.get("anchored_lexical_bias_enabled", True)
        ),
        anchored_fusion_weight_vector=float(
            patha_cfg.get("anchored_fusion_weight_vector", 0.4)
        ),
        anchored_fusion_weight_bm25=float(
            patha_cfg.get("anchored_fusion_weight_bm25", 0.6)
        ),
        fusion_weight_vector=float(patha_cfg.get("fusion_weight_vector", 0.7)),
        fusion_weight_bm25=float(patha_cfg.get("fusion_weight_bm25", 0.3)),
    )


def _expected_rank(docs: Sequence[Mapping[str, Any]], expected: Sequence[str]) -> int | None:
    expected_set = set(expected)
    if not expected_set:
        return None
    for idx, doc in enumerate(docs, start=1):
        checksum = doc.get("checksum")
        if isinstance(checksum, str) and checksum in expected_set:
            return idx
    return None


def _doc_by_checksum(
    docs: Sequence[Mapping[str, Any]],
    checksum: str | None,
) -> Mapping[str, Any] | None:
    if not checksum:
        return None
    for doc in docs:
        if doc.get("checksum") == checksum:
            return doc
    return None


def _feature_snapshot(doc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    fields = (
        "checksum",
        "path",
        "filename",
        "doc_type",
        "person_name",
        "retrieval_score",
        "score_vector",
        "score_bm25",
        "_bm25_variant_weight",
        "authority_rank",
        "_profile_intent_adjustment",
        "modified_at",
    )
    return {field: doc.get(field) for field in fields}


def _rank_value_for_delta(rank: int | None, depth: int) -> int:
    if rank is None:
        return depth + 1
    return rank


def _most_common_wrong_top1(
    failed_rows: Sequence[Mapping[str, Any]],
    metadata_cache: Mapping[str, Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in failed_rows:
        checksum = row.get("top1_checksum")
        if isinstance(checksum, str) and checksum:
            counter[checksum] += 1

    records: list[dict[str, Any]] = []
    for checksum, count in counter.most_common():
        if count < 2:
            continue
        meta = metadata_cache.get(checksum) or {}
        records.append(
            {
                "checksum": checksum,
                "count_as_wrong_top1": count,
                "path": meta.get("path"),
                "filename": meta.get("filename"),
                "doc_type": meta.get("doc_type"),
            }
        )
    return records


def _query_rows_by_id(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        query_id = row.get("query_id")
        if isinstance(query_id, str):
            out[query_id] = row
    return out


def _probe_metrics_for_mode_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    probe_depth: int,
) -> dict[str, Any]:
    rank_key = (
        "answer_support_rank_probe"
        if mode == BENCHMARK_MODE_ANSWER_SUPPORT
        else "strict_retrieval_rank_probe"
    )
    total = len(rows)
    hit1 = 0
    hit3 = 0
    rr_sum = 0.0
    rank_buckets: Counter[str] = Counter()
    for row in rows:
        rank_raw = row.get(rank_key)
        rank: int | None = (
            rank_raw
            if isinstance(rank_raw, int) and not isinstance(rank_raw, bool)
            else None
        )
        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= 3:
            hit3 += 1
        rr_sum += reciprocal_rank(rank)
        rank_buckets[rank_bucket(rank, probe_depth)] += 1
    mrr = (rr_sum / total) if total else 0.0
    return {
        "positive_total": total,
        "hit_at_1_probe": hit1,
        "hit_at_1_probe_rate": round((hit1 / total), 4) if total else 0.0,
        "hit_at_3_probe": hit3,
        "hit_at_3_probe_rate": round((hit3 / total), 4) if total else 0.0,
        "mrr_probe": round(mrr, 4),
        "rank_bucket_counts": {
            RANK_BUCKET_1: rank_buckets.get(RANK_BUCKET_1, 0),
            RANK_BUCKET_2_3: rank_buckets.get(RANK_BUCKET_2_3, 0),
            RANK_BUCKET_4_10: rank_buckets.get(RANK_BUCKET_4_10, 0),
            RANK_BUCKET_11_TO_DEPTH: rank_buckets.get(RANK_BUCKET_11_TO_DEPTH, 0),
            RANK_BUCKET_NOT_RETRIEVED: rank_buckets.get(
                RANK_BUCKET_NOT_RETRIEVED, 0
            ),
        },
    }


def probe_metrics_by_query_type(
    *,
    per_query_rows: Sequence[Mapping[str, Any]],
    mode: str,
    probe_depth: int,
) -> dict[str, dict[str, Any]]:
    by_query_type: dict[str, dict[str, Any]] = {}
    for query_type in QUERY_TYPE_ORDER:
        rows_for_type = [
            row
            for row in per_query_rows
            if row.get("benchmark_query_type") == query_type
        ]
        by_query_type[query_type] = _probe_metrics_for_mode_rows(
            rows_for_type,
            mode=mode,
            probe_depth=probe_depth,
        )
    return by_query_type


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:  # NaN
        return None
    return numeric


def _title_or_filename(doc: Mapping[str, Any] | None) -> str:
    if not doc:
        return ""
    filename = doc.get("filename")
    if isinstance(filename, str) and filename.strip():
        return filename.strip()
    path = doc.get("path")
    if isinstance(path, str) and path.strip():
        return Path(path).name
    return ""


def _title_filename_overlap(
    *,
    query_text: str,
    doc: Mapping[str, Any] | None,
) -> dict[str, Any]:
    anchors = query_anchor_tokens(query_text)
    anchor_set = set(anchors)
    title = _title_or_filename(doc)
    title_tokens = [token for token in TOKEN_RE.findall(title.lower()) if token]
    overlap_tokens = sorted(anchor_set & set(title_tokens))
    overlap_count = len(overlap_tokens)
    overlap_ratio = (overlap_count / len(anchor_set)) if anchor_set else 0.0
    return {
        "title_or_filename": title,
        "overlap_tokens": overlap_tokens,
        "overlap_count": overlap_count,
        "overlap_ratio": round(overlap_ratio, 4),
    }


def _fusion_weights_for_query(query_text: str, cfg: RetrievalConfig) -> tuple[float, float]:
    anchored = has_strong_query_anchors(query_text.strip())
    if anchored and cfg.anchored_lexical_bias_enabled:
        return cfg.anchored_fusion_weight_vector, cfg.anchored_fusion_weight_bm25
    return cfg.fusion_weight_vector, cfg.fusion_weight_bm25


def _score_contributions(
    *,
    doc: Mapping[str, Any] | None,
    fusion_weight_vector: float,
    fusion_weight_bm25: float,
) -> dict[str, float | None]:
    if not doc:
        return {
            "vector_component": None,
            "lexical_component": None,
            "doc_type_prior_component": None,
            "final_retrieval_score": None,
        }
    score_vector = _safe_float(doc.get("score_vector")) or 0.0
    score_bm25 = _safe_float(doc.get("score_bm25")) or 0.0
    bm25_variant_weight = _safe_float(doc.get("_bm25_variant_weight")) or 1.0
    vector_component = fusion_weight_vector * score_vector
    lexical_component = fusion_weight_bm25 * score_bm25 * bm25_variant_weight
    doc_type_prior_component = _safe_float(doc.get("_profile_intent_adjustment")) or 0.0
    return {
        "vector_component": round(vector_component, 6),
        "lexical_component": round(lexical_component, 6),
        "doc_type_prior_component": round(doc_type_prior_component, 6),
        "final_retrieval_score": _safe_float(doc.get("retrieval_score")),
    }


def _chunk_aggregation_signals(doc: Mapping[str, Any] | None) -> dict[str, Any]:
    if not doc:
        return {
            "source_channel": None,
            "has_vector_signal": None,
            "has_lexical_signal": None,
            "bm25_variant_weight": None,
            "chunk_id": None,
            "chunk_index": None,
            "cv_family_size": None,
            "cv_family_suppressed": None,
            "cv_family_choice_reason": None,
        }
    score_vector = _safe_float(doc.get("score_vector")) or 0.0
    score_bm25 = _safe_float(doc.get("score_bm25")) or 0.0
    return {
        "source_channel": doc.get("source"),
        "has_vector_signal": score_vector > 0.0,
        "has_lexical_signal": score_bm25 > 0.0,
        "bm25_variant_weight": _safe_float(doc.get("_bm25_variant_weight")),
        "chunk_id": doc.get("id") or doc.get("_id"),
        "chunk_index": doc.get("chunk_index"),
        "cv_family_size": doc.get("_cv_family_size"),
        "cv_family_suppressed": doc.get("_cv_family_suppressed"),
        "cv_family_choice_reason": doc.get("_cv_family_choice_reason"),
    }


def assign_primary_ranking_cause(
    *,
    expected_rank_probe: int | None,
    winner_vector_minus_expected: float | None,
    winner_lexical_minus_expected: float | None,
    title_overlap_count_delta: int | None,
    title_overlap_ratio_delta: float | None,
    doc_type_prior_delta: float | None,
    near_duplicate_collision: bool,
    chunk_aggregation_bias: bool,
) -> str:
    if expected_rank_probe is None:
        return RANKING_CAUSE_CANDIDATE_GENERATION_MISS
    if near_duplicate_collision:
        return RANKING_CAUSE_SIBLING_COLLISION
    if doc_type_prior_delta is not None and doc_type_prior_delta >= 0.02:
        return RANKING_CAUSE_DOC_TYPE_PRIOR_SUPPRESSION
    if (
        title_overlap_count_delta is not None
        and title_overlap_count_delta <= -1
        and title_overlap_ratio_delta is not None
        and title_overlap_ratio_delta <= -0.2
    ):
        return RANKING_CAUSE_TITLE_FILENAME_UNDERWEIGHTING
    if (
        winner_vector_minus_expected is not None
        and winner_vector_minus_expected >= 0.05
        and (
            winner_lexical_minus_expected is None
            or winner_lexical_minus_expected <= 0.03
        )
    ):
        return RANKING_CAUSE_VECTOR_DOMINANCE
    if chunk_aggregation_bias:
        return RANKING_CAUSE_CHUNK_AGGREGATION_BIAS
    return RANKING_CAUSE_AMBIGUOUS


def _metadata_title_or_filename(metadata: Mapping[str, Any] | None) -> str:
    if not metadata:
        return ""
    filename = metadata.get("filename")
    if isinstance(filename, str) and filename.strip():
        return filename.strip()
    path = metadata.get("path")
    if isinstance(path, str) and path.strip():
        return Path(path).name
    return ""


def _doc_family_key(
    metadata: Mapping[str, Any] | None,
    fallback_checksum: str | None,
) -> str | None:
    title = _metadata_title_or_filename(metadata)
    stem = Path(title).stem if title else ""
    tokens = [
        token
        for token in TOKEN_RE.findall(stem.lower())
        if len(token) >= 3 and not token.isdigit() and token not in GENERIC_HARD_NEGATIVE_TERMS
    ]
    if tokens:
        return "_".join(tokens[:4])
    if stem:
        compact = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
        if compact:
            return compact
    if isinstance(fallback_checksum, str) and fallback_checksum:
        return fallback_checksum
    return None


def _generic_artifact_terms_from_metadata(
    metadata: Mapping[str, Any] | None,
) -> list[str]:
    title = _metadata_title_or_filename(metadata)
    if not title:
        return []
    tokens = set(TOKEN_RE.findall(title.lower()))
    return sorted(tokens & GENERIC_HARD_NEGATIVE_TERMS)


def _is_generic_artifact_metadata(metadata: Mapping[str, Any] | None) -> bool:
    return bool(_generic_artifact_terms_from_metadata(metadata))


def _hard_negative_pattern_context(row: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics_raw = row.get("winner_vs_expected_diagnostics")
    diagnostics = diagnostics_raw if isinstance(diagnostics_raw, Mapping) else {}
    title_diag_raw = diagnostics.get("title_filename_overlap_features")
    title_diag = title_diag_raw if isinstance(title_diag_raw, Mapping) else {}
    winner_title_diag_raw = title_diag.get("winner")
    winner_title_diag = (
        winner_title_diag_raw if isinstance(winner_title_diag_raw, Mapping) else {}
    )
    expected_title_diag_raw = title_diag.get("expected")
    expected_title_diag = (
        expected_title_diag_raw if isinstance(expected_title_diag_raw, Mapping) else {}
    )

    winner_metadata_raw = row.get("winner_doc_metadata")
    winner_metadata = (
        winner_metadata_raw if isinstance(winner_metadata_raw, Mapping) else {}
    )
    expected_metadata_raw = row.get("expected_doc_metadata")
    expected_metadata = (
        expected_metadata_raw if isinstance(expected_metadata_raw, Mapping) else {}
    )
    winner_title = _metadata_title_or_filename(winner_metadata) or str(
        winner_title_diag.get("title_or_filename") or ""
    )
    expected_title = _metadata_title_or_filename(expected_metadata) or str(
        expected_title_diag.get("title_or_filename") or ""
    )
    winner_tokens = set(TOKEN_RE.findall(winner_title.lower()))
    generic_list_terms = sorted(winner_tokens & GENERIC_LIST_INDEX_TERMS)
    aggregate_summary_terms = sorted(winner_tokens & GENERIC_AGGREGATE_SUMMARY_TERMS)

    vector_diag_raw = diagnostics.get("vector_score_contribution")
    vector_diag = vector_diag_raw if isinstance(vector_diag_raw, Mapping) else {}
    lexical_diag_raw = diagnostics.get("lexical_score_contribution")
    lexical_diag = lexical_diag_raw if isinstance(lexical_diag_raw, Mapping) else {}
    vector_delta = _safe_float(vector_diag.get("delta_winner_minus_expected"))
    lexical_delta = _safe_float(lexical_diag.get("delta_winner_minus_expected"))

    winner_overlap_count = int(winner_title_diag.get("overlap_count") or 0)
    expected_overlap_count = int(expected_title_diag.get("overlap_count") or 0)
    weak_title_vector_signal = (
        winner_overlap_count <= 0
        and expected_overlap_count >= 1
        and vector_delta is not None
        and vector_delta >= 0.05
        and (lexical_delta is None or lexical_delta <= 0.05)
    )

    winner_checksum = (
        str(row.get("actual_top1_checksum"))
        if isinstance(row.get("actual_top1_checksum"), str)
        else None
    )
    expected_checksum = (
        str(row.get("expected_checksum"))
        if isinstance(row.get("expected_checksum"), str)
        else None
    )
    winner_family_key = _doc_family_key(winner_metadata, winner_checksum)
    expected_family_key = _doc_family_key(expected_metadata, expected_checksum)

    similarity_raw = diagnostics.get("winner_expected_text_similarity")
    similarity = similarity_raw if isinstance(similarity_raw, Mapping) else {}
    near_duplicate_collision = bool(similarity.get("near_duplicate"))
    family_collision = bool(
        winner_family_key
        and expected_family_key
        and winner_family_key == expected_family_key
        and winner_checksum
        and expected_checksum
        and winner_checksum != expected_checksum
    )
    sibling_collision = near_duplicate_collision or family_collision

    pattern_classes: list[str] = []
    if generic_list_terms:
        pattern_classes.append(HARD_NEGATIVE_CLASS_GENERIC_LIST_INDEX)
    if aggregate_summary_terms:
        pattern_classes.append(HARD_NEGATIVE_CLASS_AGGREGATE_SUMMARY)
    if sibling_collision:
        pattern_classes.append(HARD_NEGATIVE_CLASS_SIBLING_FAMILY_COLLISION)
    if weak_title_vector_signal:
        pattern_classes.append(HARD_NEGATIVE_CLASS_WEAK_TITLE_VECTOR)

    return {
        "pattern_classes": pattern_classes,
        "winner_family_key": winner_family_key,
        "expected_family_key": expected_family_key,
        "winner_generic_list_terms": generic_list_terms,
        "winner_aggregate_summary_terms": aggregate_summary_terms,
        "near_duplicate_collision": near_duplicate_collision,
        "family_collision": family_collision,
        "weak_title_vector_signal": weak_title_vector_signal,
    }


def analyze_strict_canonical_hard_negatives(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    annotated_rows: list[dict[str, Any]] = []
    winner_checksum_counter: Counter[str] = Counter()
    winner_family_counter: Counter[str] = Counter()
    pattern_counter: Counter[str] = Counter()

    for row in rows:
        context = _hard_negative_pattern_context(row)
        winner_checksum = row.get("actual_top1_checksum")
        if isinstance(winner_checksum, str) and winner_checksum:
            winner_checksum_counter[winner_checksum] += 1
        winner_family_key = context.get("winner_family_key")
        if isinstance(winner_family_key, str) and winner_family_key:
            winner_family_counter[winner_family_key] += 1
        for pattern_class in context.get("pattern_classes", []):
            if isinstance(pattern_class, str):
                pattern_counter[pattern_class] += 1

        winner_metadata_raw = row.get("winner_doc_metadata")
        winner_metadata = (
            winner_metadata_raw if isinstance(winner_metadata_raw, Mapping) else {}
        )
        annotated_rows.append(
            {
                "query_id": row.get("query_id"),
                "query": row.get("query"),
                "primary_ranking_cause_bucket": row.get("primary_ranking_cause_bucket"),
                "actual_top1_checksum": winner_checksum,
                "expected_checksum": row.get("expected_checksum"),
                "winner_path": winner_metadata.get("path"),
                "winner_filename": winner_metadata.get("filename"),
                "winner_doc_type": winner_metadata.get("doc_type"),
                "winner_family_key": context.get("winner_family_key"),
                "hard_negative_pattern_classes": context.get("pattern_classes", []),
                "winner_generic_list_terms": context.get("winner_generic_list_terms", []),
                "winner_aggregate_summary_terms": context.get(
                    "winner_aggregate_summary_terms", []
                ),
                "weak_title_vector_signal": bool(
                    context.get("weak_title_vector_signal")
                ),
                "sibling_family_collision": bool(
                    context.get("family_collision")
                    or context.get("near_duplicate_collision")
                ),
            }
        )

    repeated_wrong_winner_docs: list[dict[str, Any]] = []
    for row in annotated_rows:
        checksum = row.get("actual_top1_checksum")
        if not isinstance(checksum, str) or winner_checksum_counter.get(checksum, 0) < 2:
            continue
        repeated_wrong_winner_docs.append(
            {
                "checksum": checksum,
                "count": winner_checksum_counter[checksum],
                "path": row.get("winner_path"),
                "filename": row.get("winner_filename"),
                "doc_type": row.get("winner_doc_type"),
            }
        )
    repeated_wrong_winner_docs = sorted(
        {
            (record["checksum"], record["count"]): record
            for record in repeated_wrong_winner_docs
        }.values(),
        key=lambda item: (-int(item.get("count", 0)), str(item.get("checksum") or "")),
    )

    repeated_wrong_winner_families: list[dict[str, Any]] = []
    for family_key, count in winner_family_counter.items():
        if count < 2:
            continue
        repeated_wrong_winner_families.append(
            {
                "family_key": family_key,
                "count": count,
            }
        )
    repeated_wrong_winner_families.sort(
        key=lambda item: (-int(item.get("count", 0)), str(item.get("family_key") or ""))
    )

    likely_ranking_fix_candidates: list[dict[str, Any]] = []
    likely_benchmark_ambiguity_candidates: list[dict[str, Any]] = []
    for row in annotated_rows:
        cause_bucket = str(row.get("primary_ranking_cause_bucket") or "")
        pattern_classes = {
            str(item)
            for item in row.get("hard_negative_pattern_classes", [])
            if isinstance(item, str)
        }
        checksum = row.get("actual_top1_checksum")
        repeat_count = (
            winner_checksum_counter.get(checksum, 0)
            if isinstance(checksum, str) and checksum
            else 0
        )
        repeated_generic_winner = repeat_count >= 2 and bool(
            pattern_classes
            & {
                HARD_NEGATIVE_CLASS_GENERIC_LIST_INDEX,
                HARD_NEGATIVE_CLASS_AGGREGATE_SUMMARY,
            }
        )
        likely_ranking_fix = (
            repeated_generic_winner
            or HARD_NEGATIVE_CLASS_WEAK_TITLE_VECTOR in pattern_classes
            or HARD_NEGATIVE_CLASS_SIBLING_FAMILY_COLLISION in pattern_classes
            or cause_bucket
            in {
                RANKING_CAUSE_VECTOR_DOMINANCE,
                RANKING_CAUSE_TITLE_FILENAME_UNDERWEIGHTING,
                RANKING_CAUSE_CANDIDATE_GENERATION_MISS,
                RANKING_CAUSE_CHUNK_AGGREGATION_BIAS,
                RANKING_CAUSE_DOC_TYPE_PRIOR_SUPPRESSION,
            }
        )
        review_bucket = (
            LIKELY_RANKING_FIX_CANDIDATE
            if likely_ranking_fix
            else LIKELY_BENCHMARK_AMBIGUITY
        )
        row["winner_repeat_count"] = repeat_count
        row["evidence_bucket"] = review_bucket
        row["benchmark_label_ambiguity_flag"] = review_bucket == LIKELY_BENCHMARK_AMBIGUITY
        if review_bucket == LIKELY_RANKING_FIX_CANDIDATE:
            likely_ranking_fix_candidates.append(row)
        else:
            likely_benchmark_ambiguity_candidates.append(row)

    repeated_generic_pattern_detected = any(
        (
            record.get("count", 0) >= 2
            and any(
                str(row.get("actual_top1_checksum") or "") == str(record.get("checksum") or "")
                and bool(
                    {
                        str(item)
                        for item in row.get("hard_negative_pattern_classes", [])
                        if isinstance(item, str)
                    }
                    & {
                        HARD_NEGATIVE_CLASS_GENERIC_LIST_INDEX,
                        HARD_NEGATIVE_CLASS_AGGREGATE_SUMMARY,
                    }
                )
                for row in annotated_rows
            )
        )
        for record in repeated_wrong_winner_docs
    )

    return {
        "scope": (
            "strict canonical ranking misses only "
            "(benchmark_query_type=canonical_document_query and strict hit@1 miss)"
        ),
        "summary": {
            "rows_analyzed": len(annotated_rows),
            "pattern_counts": {
                pattern: int(pattern_counter.get(pattern, 0))
                for pattern in HARD_NEGATIVE_CLASS_ORDER
            },
            "repeated_wrong_winner_docs": len(repeated_wrong_winner_docs),
            "repeated_wrong_winner_families": len(repeated_wrong_winner_families),
            "likely_ranking_fix_candidates": len(likely_ranking_fix_candidates),
            "likely_benchmark_ambiguity_manual_review_candidates": len(
                likely_benchmark_ambiguity_candidates
            ),
        },
        "suppression_signal": {
            "repeated_generic_hard_negative_pattern_detected": repeated_generic_pattern_detected,
            "candidate_rule_scope": "canonical anchored/semi-anchored queries only",
        },
        "repeated_wrong_winner_docs": repeated_wrong_winner_docs,
        "repeated_wrong_winner_families": repeated_wrong_winner_families,
        "likely_ranking_fix_candidates": likely_ranking_fix_candidates,
        "likely_benchmark_ambiguity_manual_review_candidates": likely_benchmark_ambiguity_candidates,
        "rows": annotated_rows,
    }


def build_strict_canonical_cleaned_residual_split(
    *,
    archived_rows: Sequence[Mapping[str, Any]],
    per_query_rows: Sequence[Mapping[str, Any]],
    strict_canonical_rows: Sequence[Mapping[str, Any]],
    strict_canonical_hard_negative_analysis: Mapping[str, Any],
    metadata_lookup: Any,
    artifact_near_tie_epsilon: float = DEFAULT_ARTIFACT_NEAR_TIE_EPSILON,
) -> dict[str, Any]:
    per_query_by_id = _query_rows_by_id(per_query_rows)
    strict_row_by_id: dict[str, Mapping[str, Any]] = {}
    for row in strict_canonical_rows:
        query_id = row.get("query_id")
        if isinstance(query_id, str) and query_id:
            strict_row_by_id[query_id] = row

    hard_negative_rows = strict_canonical_hard_negative_analysis.get("rows", [])
    hard_negative_row_by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(hard_negative_rows, Sequence):
        for row in hard_negative_rows:
            if not isinstance(row, Mapping):
                continue
            query_id = row.get("query_id")
            if isinstance(query_id, str) and query_id:
                hard_negative_row_by_id[query_id] = row

    strict_canonical_benchmark_failure_query_ids: list[str] = []
    for row in per_query_rows:
        if not isinstance(row, Mapping):
            continue
        if row.get("benchmark_query_type") != QUERY_TYPE_CANONICAL_DOCUMENT:
            continue
        if row.get("benchmark_primary_mode") != BENCHMARK_MODE_STRICT_RETRIEVAL:
            continue
        if not bool(row.get("selected_for_ranking_failure_analysis")):
            continue
        query_id = row.get("query_id")
        if isinstance(query_id, str) and query_id:
            strict_canonical_benchmark_failure_query_ids.append(query_id)

    actionable_rows: list[dict[str, Any]] = []
    ambiguity_rows: list[dict[str, Any]] = []
    actionable_bucket_counter: Counter[str] = Counter()

    for query_id in strict_canonical_benchmark_failure_query_ids:
        cause_row = strict_row_by_id.get(query_id)
        if not cause_row:
            continue
        hard_row = hard_negative_row_by_id.get(query_id, {})
        cause_bucket = str(cause_row.get("primary_ranking_cause_bucket") or "")
        evidence_bucket = str(hard_row.get("evidence_bucket") or "").strip()
        if evidence_bucket not in {
            LIKELY_RANKING_FIX_CANDIDATE,
            LIKELY_BENCHMARK_AMBIGUITY,
        }:
            evidence_bucket = (
                LIKELY_BENCHMARK_AMBIGUITY
                if cause_bucket == RANKING_CAUSE_AMBIGUOUS
                else LIKELY_RANKING_FIX_CANDIDATE
            )

        row_payload = {
            "query_id": query_id,
            "query": cause_row.get("query"),
            "primary_ranking_cause_bucket": cause_bucket,
            "evidence_bucket": evidence_bucket,
            "hard_negative_pattern_classes": (
                list(hard_row.get("hard_negative_pattern_classes", []))
                if isinstance(hard_row.get("hard_negative_pattern_classes"), Sequence)
                and not isinstance(hard_row.get("hard_negative_pattern_classes"), str)
                else []
            ),
            "expected_rank_if_retrieved": cause_row.get("expected_rank_if_retrieved"),
            "actual_top1_checksum": cause_row.get("actual_top1_checksum"),
            "expected_checksum": cause_row.get("expected_checksum"),
        }
        if evidence_bucket == LIKELY_BENCHMARK_AMBIGUITY:
            ambiguity_rows.append(row_payload)
            continue
        actionable_rows.append(row_payload)
        if cause_bucket:
            actionable_bucket_counter[cause_bucket] += 1

    actionable_bucket_counts = {
        bucket: int(actionable_bucket_counter.get(bucket, 0))
        for bucket in RANKING_CAUSE_ORDER
        if bucket != RANKING_CAUSE_AMBIGUOUS
    }
    largest_actionable_bucket: str | None = None
    largest_actionable_bucket_count = 0
    for bucket in RANKING_CAUSE_ORDER:
        if bucket == RANKING_CAUSE_AMBIGUOUS:
            continue
        count = int(actionable_bucket_counts.get(bucket, 0))
        if count > largest_actionable_bucket_count:
            largest_actionable_bucket = bucket
            largest_actionable_bucket_count = count

    largest_actionable_representative_query_ids: list[str] = []
    if largest_actionable_bucket:
        largest_actionable_representative_query_ids = [
            str(row.get("query_id"))
            for row in actionable_rows
            if row.get("primary_ranking_cause_bucket") == largest_actionable_bucket
            and isinstance(row.get("query_id"), str)
        ][:5]

    already_addressed_artifact_first_rows: list[dict[str, Any]] = []
    for archived_row in archived_rows:
        query_id = archived_row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            continue
        diag = per_query_by_id.get(query_id, {})
        if (
            diag.get("benchmark_query_type") != QUERY_TYPE_CANONICAL_DOCUMENT
            or diag.get("benchmark_primary_mode") != BENCHMARK_MODE_STRICT_RETRIEVAL
        ):
            continue
        if not bool(archived_row.get("hit_at_1")):
            continue

        top1_score = _safe_float(archived_row.get("top1_score"))
        if top1_score is None:
            continue
        top1_checksum_raw = archived_row.get("top1_checksum")
        top1_checksum = (
            str(top1_checksum_raw)
            if isinstance(top1_checksum_raw, str) and top1_checksum_raw
            else None
        )
        top1_meta = (
            metadata_lookup(top1_checksum) if callable(metadata_lookup) and top1_checksum else None
        )
        artifact_competitors: list[dict[str, Any]] = []
        for rank in (2, 3):
            checksum_raw = archived_row.get(f"top{rank}_checksum")
            checksum = (
                str(checksum_raw)
                if isinstance(checksum_raw, str) and checksum_raw
                else None
            )
            score = _safe_float(archived_row.get(f"top{rank}_score"))
            if checksum is None or score is None:
                continue
            score_delta_vs_top1 = top1_score - score
            if (
                score_delta_vs_top1 < 0.0
                or score_delta_vs_top1 > artifact_near_tie_epsilon
            ):
                continue
            metadata = metadata_lookup(checksum) if callable(metadata_lookup) else None
            if not _is_generic_artifact_metadata(metadata):
                continue
            artifact_competitors.append(
                {
                    "rank": rank,
                    "checksum": checksum,
                    "score": score,
                    "score_delta_vs_top1": round(score_delta_vs_top1, 6),
                    "path": (metadata or {}).get("path"),
                    "filename": (metadata or {}).get("filename"),
                    "doc_type": (metadata or {}).get("doc_type"),
                    "artifact_terms": _generic_artifact_terms_from_metadata(metadata),
                }
            )
        if not artifact_competitors:
            continue
        already_addressed_artifact_first_rows.append(
            {
                "query_id": query_id,
                "query": archived_row.get("query"),
                "top1_checksum": top1_checksum,
                "top1_score": top1_score,
                "top1_path": (top1_meta or {}).get("path"),
                "top1_filename": (top1_meta or {}).get("filename"),
                "top1_doc_type": (top1_meta or {}).get("doc_type"),
                "artifact_competitors_within_near_tie_epsilon": artifact_competitors,
            }
        )

    already_addressed_artifact_first_rows.sort(
        key=lambda row: str(row.get("query_id") or "")
    )
    actionable_rows.sort(key=lambda row: str(row.get("query_id") or ""))
    ambiguity_rows.sort(key=lambda row: str(row.get("query_id") or ""))

    return {
        "scope": (
            "benchmark-cleaned strict canonical residual split "
            "(benchmark_query_type=canonical_document_query, benchmark_primary_mode=strict_retrieval)"
        ),
        "summary": {
            "strict_canonical_benchmark_failures_total": len(
                strict_canonical_benchmark_failure_query_ids
            ),
            "actionable_ranking_failures": len(actionable_rows),
            "likely_benchmark_ambiguity_manual_review": len(ambiguity_rows),
            "already_addressed_artifact_first_cases": len(
                already_addressed_artifact_first_rows
            ),
            "artifact_near_tie_epsilon": artifact_near_tie_epsilon,
        },
        "actionable_ranking_failures": {
            "query_ids": [row.get("query_id") for row in actionable_rows],
            "bucket_counts": actionable_bucket_counts,
            "largest_bucket": {
                "bucket": largest_actionable_bucket,
                "count": largest_actionable_bucket_count,
                "representative_query_ids": largest_actionable_representative_query_ids,
            },
            "rows": actionable_rows,
        },
        "likely_benchmark_ambiguity_manual_review": {
            "query_ids": [row.get("query_id") for row in ambiguity_rows],
            "rows": ambiguity_rows,
        },
        "already_addressed_artifact_first_cases": {
            "query_ids": [
                row.get("query_id") for row in already_addressed_artifact_first_rows
            ],
            "rows": already_addressed_artifact_first_rows,
        },
    }


def _chunk_document_collapsing_profile(cfg: RetrievalConfig) -> dict[str, Any]:
    return {
        "checksum_dedup_enabled": True,
        "cv_family_collapse_enabled": cfg.cv_family_collapse_enabled,
        "near_duplicate_collapse_enabled": cfg.sim_threshold > 0,
        "near_duplicate_similarity_threshold": cfg.sim_threshold,
        "mmr_enabled": cfg.enable_mmr,
        "mmr_effective_k": cfg.mmr_k or cfg.top_k,
        "duplicate_topup_enabled": cfg.include_dups_if_needed,
    }


def _artifact_method_profile(
    *,
    artifact_path: Path,
    variants_enabled: bool,
    rewrites_enabled: bool,
    exact_query_probing: bool,
    candidate_depth: int,
    top_k_each: int | None,
    fusion_configuration: Mapping[str, Any],
    chunk_document_collapsing: Mapping[str, Any],
    metric_interpretation_scope: str,
) -> dict[str, Any]:
    return {
        "artifact_path": str(artifact_path),
        "variants_enabled": variants_enabled,
        "rewrites_enabled": rewrites_enabled,
        "exact_query_probing": exact_query_probing,
        "candidate_depth": candidate_depth,
        "top_k_each": top_k_each,
        "fusion_configuration": dict(fusion_configuration),
        "chunk_document_collapsing_behavior": dict(chunk_document_collapsing),
        "metric_interpretation_scope": metric_interpretation_scope,
    }


def build_probe_vs_eval_comparison(
    *,
    patha_runbook_path: Path,
    ranking_artifact_path: Path,
    runbook: Mapping[str, Any],
    archived_rows: Sequence[Mapping[str, Any]],
    per_query_rows: Sequence[Mapping[str, Any]],
    probe_docs_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
    probe_cfg: RetrievalConfig,
) -> dict[str, Any]:
    runbook_cfg_raw = runbook.get("config", {}) or {}
    runbook_cfg = runbook_cfg_raw if isinstance(runbook_cfg_raw, Mapping) else {}
    default_cfg = RetrievalConfig()
    eval_candidate_depth = int(runbook_cfg.get("top_k") or 3)
    eval_cfg = RetrievalConfig(
        top_k=max(eval_candidate_depth, 1),
        enable_variants=bool(runbook_cfg.get("enable_variants", True)),
        enable_mmr=bool(runbook_cfg.get("enable_mmr", True)),
        anchored_exact_only=bool(runbook_cfg.get("anchored_exact_only", True)),
        anchored_lexical_bias_enabled=bool(
            runbook_cfg.get("anchored_lexical_bias_enabled", True)
        ),
        anchored_fusion_weight_vector=float(
            runbook_cfg.get(
                "anchored_fusion_weight_vector",
                default_cfg.anchored_fusion_weight_vector,
            )
        ),
        anchored_fusion_weight_bm25=float(
            runbook_cfg.get(
                "anchored_fusion_weight_bm25",
                default_cfg.anchored_fusion_weight_bm25,
            )
        ),
        fusion_weight_vector=float(
            runbook_cfg.get("fusion_weight_vector", default_cfg.fusion_weight_vector)
        ),
        fusion_weight_bm25=float(
            runbook_cfg.get("fusion_weight_bm25", default_cfg.fusion_weight_bm25)
        ),
    )
    eval_profile = _artifact_method_profile(
        artifact_path=patha_runbook_path,
        variants_enabled=eval_cfg.enable_variants,
        rewrites_enabled=eval_cfg.enable_variants,
        exact_query_probing=False,
        candidate_depth=eval_cfg.top_k,
        top_k_each=None,
        fusion_configuration={
            "anchored_exact_only": eval_cfg.anchored_exact_only,
            "anchored_lexical_bias_enabled": eval_cfg.anchored_lexical_bias_enabled,
            "fusion_weight_vector": eval_cfg.fusion_weight_vector,
            "fusion_weight_bm25": eval_cfg.fusion_weight_bm25,
            "anchored_fusion_weight_vector": eval_cfg.anchored_fusion_weight_vector,
            "anchored_fusion_weight_bm25": eval_cfg.anchored_fusion_weight_bm25,
        },
        chunk_document_collapsing=_chunk_document_collapsing_profile(eval_cfg),
        metric_interpretation_scope=(
            "Archived strict metrics from runbook summary "
            "(positive_hit_at_1/positive_hit_at_3 over mode=positive rows, "
            "strict expected-checksum match within top_k window)."
        ),
    )
    probe_profile = _artifact_method_profile(
        artifact_path=ranking_artifact_path,
        variants_enabled=probe_cfg.enable_variants,
        rewrites_enabled=probe_cfg.enable_variants,
        exact_query_probing=True,
        candidate_depth=probe_cfg.top_k,
        top_k_each=probe_cfg.top_k_each,
        fusion_configuration={
            "anchored_exact_only": probe_cfg.anchored_exact_only,
            "anchored_lexical_bias_enabled": probe_cfg.anchored_lexical_bias_enabled,
            "fusion_weight_vector": probe_cfg.fusion_weight_vector,
            "fusion_weight_bm25": probe_cfg.fusion_weight_bm25,
            "anchored_fusion_weight_vector": probe_cfg.anchored_fusion_weight_vector,
            "anchored_fusion_weight_bm25": probe_cfg.anchored_fusion_weight_bm25,
        },
        chunk_document_collapsing=_chunk_document_collapsing_profile(probe_cfg),
        metric_interpretation_scope=(
            "Deterministic strict probe metrics "
            "(strict_retrieval_rank_probe over exact-query deep probe depth)."
        ),
    )

    runbook_summary_raw = runbook.get("summary", {}) or {}
    runbook_summary = runbook_summary_raw if isinstance(runbook_summary_raw, Mapping) else {}
    archived_total = int(
        runbook_summary.get("positive_total")
        or len([row for row in archived_rows if row.get("mode") == "positive"])
    )
    archived_hit1 = int(runbook_summary.get("positive_hit_at_1") or 0)
    archived_hit3 = int(runbook_summary.get("positive_hit_at_3") or 0)

    diag_by_id = _query_rows_by_id(per_query_rows)
    disagreements: list[dict[str, Any]] = []
    archived_only_hit1: list[str] = []
    archived_only_hit3: list[str] = []
    probe_only_hit1: list[str] = []
    probe_only_hit3: list[str] = []
    probe_hit1 = 0
    probe_hit3 = 0

    for archived_row in archived_rows:
        query_id = str(archived_row.get("query_id") or "")
        if not query_id:
            continue
        diag_row = diag_by_id.get(query_id, {})
        archived_h1 = bool(archived_row.get("hit_at_1"))
        archived_h3 = bool(archived_row.get("hit_at_3"))
        probe_h1 = bool(diag_row.get("strict_retrieval_hit_at_1_probe"))
        probe_h3 = bool(diag_row.get("strict_retrieval_hit_at_3_probe"))
        if probe_h1:
            probe_hit1 += 1
        if probe_h3:
            probe_hit3 += 1

        if archived_h1 and not probe_h1:
            archived_only_hit1.append(query_id)
        if archived_h3 and not probe_h3:
            archived_only_hit3.append(query_id)
        if probe_h1 and not archived_h1:
            probe_only_hit1.append(query_id)
        if probe_h3 and not archived_h3:
            probe_only_hit3.append(query_id)

        if archived_h1 == probe_h1 and archived_h3 == probe_h3:
            continue

        archived_top3 = [
            checksum
            for checksum in (
                archived_row.get("top1_checksum"),
                archived_row.get("top2_checksum"),
                archived_row.get("top3_checksum"),
            )
            if isinstance(checksum, str) and checksum
        ]
        probe_top3 = [
            checksum
            for checksum in (
                doc.get("checksum") for doc in (probe_docs_by_query.get(query_id) or [])[:3]
            )
            if isinstance(checksum, str) and checksum
        ]
        disagreements.append(
            {
                "query_id": query_id,
                "query": archived_row.get("query"),
                "archived_hit_at_1": archived_h1,
                "archived_hit_at_3": archived_h3,
                "probe_hit_at_1": probe_h1,
                "probe_hit_at_3": probe_h3,
                "archived_top3_checksums": archived_top3,
                "probe_top3_checksums": probe_top3,
                "strict_retrieval_rank_probe": diag_row.get("strict_retrieval_rank_probe"),
            }
        )

    strict_metric_comparison = {
        "archived_eval": {
            "positive_total": archived_total,
            "hit_at_1": archived_hit1,
            "hit_at_1_rate": round((archived_hit1 / archived_total), 4)
            if archived_total
            else 0.0,
            "hit_at_3": archived_hit3,
            "hit_at_3_rate": round((archived_hit3 / archived_total), 4)
            if archived_total
            else 0.0,
        },
        "deterministic_probe": {
            "positive_total": archived_total,
            "hit_at_1": probe_hit1,
            "hit_at_1_rate": round((probe_hit1 / archived_total), 4)
            if archived_total
            else 0.0,
            "hit_at_3": probe_hit3,
            "hit_at_3_rate": round((probe_hit3 / archived_total), 4)
            if archived_total
            else 0.0,
        },
        "delta_probe_minus_archived": {
            "hit_at_1": probe_hit1 - archived_hit1,
            "hit_at_3": probe_hit3 - archived_hit3,
        },
    }

    explanation = (
        f"Archived Path A eval uses variants_enabled={eval_cfg.enable_variants}, "
        f"rewrites_enabled={eval_cfg.enable_variants}, exact_query_probing=False, "
        f"candidate_depth={eval_cfg.top_k}; deterministic ranking probe uses "
        f"variants_enabled={probe_cfg.enable_variants}, rewrites_enabled={probe_cfg.enable_variants}, "
        f"exact_query_probing=True, candidate_depth={probe_cfg.top_k}. "
        "retrieve() binds MMR selection depth to top_k (mmr_k defaults to top_k), "
        "so changing candidate_depth changes top-3 ordering. "
        f"Observed strict discrepancy: archived hit@1={archived_hit1}/{archived_total}, "
        f"hit@3={archived_hit3}/{archived_total} vs probe hit@1={probe_hit1}/{archived_total}, "
        f"hit@3={probe_hit3}/{archived_total}."
    )

    return {
        "artifact_method_profiles": [eval_profile, probe_profile],
        "strict_metric_comparison": strict_metric_comparison,
        "query_disagreement_summary": {
            "queries_with_any_disagreement": len(disagreements),
            "archived_only_hit_at_1_queries": len(archived_only_hit1),
            "archived_only_hit_at_3_queries": len(archived_only_hit3),
            "probe_only_hit_at_1_queries": len(probe_only_hit1),
            "probe_only_hit_at_3_queries": len(probe_only_hit3),
            "archived_only_hit_at_1_query_ids": archived_only_hit1,
            "archived_only_hit_at_3_query_ids": archived_only_hit3,
            "probe_only_hit_at_1_query_ids": probe_only_hit1,
            "probe_only_hit_at_3_query_ids": probe_only_hit3,
        },
        "query_disagreements": disagreements,
        "deterministic_explanation": explanation,
    }


def _best_similarity_to_expected(
    *,
    doc_checksum: str,
    doc_text: str,
    expected_checksums: Sequence[str],
    expected_text_by_checksum: Mapping[str, str],
    similarity_cache: dict[tuple[str, str], dict[str, float | bool]],
) -> tuple[str | None, dict[str, float | bool] | None]:
    best_expected_checksum: str | None = None
    best_similarity: dict[str, float | bool] | None = None
    for expected_checksum in expected_checksums:
        expected_text = expected_text_by_checksum.get(expected_checksum) or ""
        if not expected_text:
            continue
        cache_key = (doc_checksum, expected_checksum)
        if cache_key not in similarity_cache:
            similarity_cache[cache_key] = text_similarity_metrics(doc_text, expected_text)
        similarity = similarity_cache[cache_key]
        if best_similarity is None:
            best_similarity = similarity
            best_expected_checksum = expected_checksum
            continue
        if float(similarity.get("containment_min", 0.0) or 0.0) > float(
            best_similarity.get("containment_min", 0.0) or 0.0
        ):
            best_similarity = similarity
            best_expected_checksum = expected_checksum
    return best_expected_checksum, best_similarity


def _best_query_conditioned_similarity_to_expected(
    *,
    query_text: str,
    doc_text: str,
    expected_checksums: Sequence[str],
    expected_text_by_checksum: Mapping[str, str],
) -> tuple[str | None, dict[str, Any] | None]:
    best_expected_checksum: str | None = None
    best_similarity: dict[str, Any] | None = None
    for expected_checksum in expected_checksums:
        expected_text = expected_text_by_checksum.get(expected_checksum) or ""
        if not expected_text:
            continue
        similarity = query_conditioned_similarity_metrics(
            query=query_text,
            candidate_text=doc_text,
            expected_text=expected_text,
        )
        if not similarity:
            continue
        if best_similarity is None:
            best_similarity = similarity
            best_expected_checksum = expected_checksum
            continue
        if float(similarity.get("containment_min", 0.0) or 0.0) > float(
            best_similarity.get("containment_min", 0.0) or 0.0
        ):
            best_similarity = similarity
            best_expected_checksum = expected_checksum
    return best_expected_checksum, best_similarity


def find_first_answer_support(
    *,
    docs: Sequence[Mapping[str, Any]],
    expected_checksums: Sequence[str],
    metadata_cache: Mapping[str, Mapping[str, Any] | None],
    similarity_cache: dict[tuple[str, str], dict[str, float | bool]],
    allow_equivalent: bool = False,
) -> dict[str, Any]:
    expected_set = set(expected_checksums)
    expected_text_by_checksum: dict[str, str] = {}
    if allow_equivalent:
        for checksum in expected_checksums:
            meta = metadata_cache.get(checksum) or {}
            text = str(meta.get("text_full") or "")
            if text:
                expected_text_by_checksum[checksum] = text

    for rank, doc in enumerate(docs, start=1):
        checksum = doc.get("checksum")
        if not isinstance(checksum, str) or not checksum:
            continue
        if checksum in expected_set:
            return {
                "rank": rank,
                "checksum": checksum,
                "support_type": "strict",
                "matched_expected_checksum": checksum,
                "similarity": None,
            }
        if not allow_equivalent:
            continue
        meta = metadata_cache.get(checksum) or {}
        text = str(meta.get("text_full") or "")
        if not text:
            continue
        matched_expected, similarity = _best_similarity_to_expected(
            doc_checksum=checksum,
            doc_text=text,
            expected_checksums=expected_checksums,
            expected_text_by_checksum=expected_text_by_checksum,
            similarity_cache=similarity_cache,
        )
        if is_equivalent_answer_support(similarity):
            return {
                "rank": rank,
                "checksum": checksum,
                "support_type": "equivalent",
                "matched_expected_checksum": matched_expected,
                "similarity": similarity,
            }

    return {
        "rank": None,
        "checksum": None,
        "support_type": None,
        "matched_expected_checksum": None,
        "similarity": None,
    }


def build_answer_support_review_candidates(
    *,
    query_text: str,
    docs: Sequence[Mapping[str, Any]],
    support_expected_checksums: Sequence[str],
    metadata_cache: Mapping[str, Mapping[str, Any] | None],
    similarity_cache: dict[tuple[str, str], dict[str, float | bool]],
    rank_limit: int,
) -> list[dict[str, Any]]:
    expected_set = set(support_expected_checksums)
    expected_text_by_checksum: dict[str, str] = {}
    for checksum in support_expected_checksums:
        meta = metadata_cache.get(checksum) or {}
        text = str(meta.get("text_full") or "")
        if text:
            expected_text_by_checksum[checksum] = text

    out: list[dict[str, Any]] = []
    for rank, doc in enumerate(docs, start=1):
        if rank > rank_limit:
            break
        checksum = doc.get("checksum")
        if not isinstance(checksum, str) or not checksum or checksum in expected_set:
            continue
        meta = metadata_cache.get(checksum) or {}
        text = str(meta.get("text_full") or "")
        if not text:
            continue

        best_expected_checksum, best_similarity = _best_similarity_to_expected(
            doc_checksum=checksum,
            doc_text=text,
            expected_checksums=support_expected_checksums,
            expected_text_by_checksum=expected_text_by_checksum,
            similarity_cache=similarity_cache,
        )
        query_best_expected_checksum, query_similarity = _best_query_conditioned_similarity_to_expected(
            query_text=query_text,
            doc_text=text,
            expected_checksums=support_expected_checksums,
            expected_text_by_checksum=expected_text_by_checksum,
        )

        review_reason: str | None = None
        if is_equivalent_answer_support(best_similarity):
            review_reason = "equivalent_fulltext_threshold"
        elif is_review_candidate_similarity(query_similarity):
            review_reason = "query_conditioned_similarity_threshold"
        if review_reason is None:
            continue

        out.append(
            {
                "rank": rank,
                "checksum": checksum,
                "path": meta.get("path"),
                "doc_type": meta.get("doc_type"),
                "review_reason": review_reason,
                "fulltext_best_expected_checksum": best_expected_checksum,
                "fulltext_similarity": best_similarity,
                "query_conditioned_best_expected_checksum": query_best_expected_checksum,
                "query_conditioned_similarity": query_similarity,
                "reviewer_mark_answer_support": None,
                "reviewer_notes": "",
            }
        )
    return out


def investigate_ranking_post_patha(
    *,
    patha_runbook_path: Path,
    fixture_path: Path,
    output_path: Path,
    probe_depth: int,
    support_labels_path: Path | None = DEFAULT_SUPPORT_LABELS_PATH,
    review_rank_limit: int = DEFAULT_REVIEW_RANK_LIMIT,
    cleaned_strict_output_path: Path | None = None,
    artifact_near_tie_epsilon: float = DEFAULT_ARTIFACT_NEAR_TIE_EPSILON,
) -> dict[str, Any]:
    runbook = _load_json(patha_runbook_path)
    fixture = _load_json(fixture_path)
    fixture_rows = fixture.get("queries", [])
    fixture_by_id = {
        str(row.get("id")): row
        for row in fixture_rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    support_label_meta, support_label_overrides = _load_support_labels(support_labels_path)
    default_positive_query_type = _normalize_query_type(
        support_label_meta.get("default_positive_query_type")
    ) or DEFAULT_POSITIVE_QUERY_TYPE
    rows = [
        row
        for row in runbook.get("rows", [])
        if isinstance(row, dict) and row.get("mode") == "positive"
    ]
    row_by_id = _query_rows_by_id(rows)
    os_client = get_client()
    metadata_cache: dict[str, dict[str, Any] | None] = {}
    similarity_cache: dict[tuple[str, str], dict[str, float | bool]] = {}

    def cached_fulltext(checksum: str) -> dict[str, Any] | None:
        if checksum not in metadata_cache:
            metadata_cache[checksum] = _fetch_fulltext_by_checksum(os_client, checksum)
        return metadata_cache[checksum]

    base_cfg = _build_probe_cfg(runbook.get("config", {}), probe_depth=probe_depth)
    ablation_cfgs = {
        "base": base_cfg,
        "no_authority": replace(base_cfg, authority_boost_enabled=False),
        "no_recency": replace(base_cfg, recency_boost_enabled=False),
        "no_profile_intent": replace(base_cfg, profile_intent_boost_enabled=False),
        "no_boosts": replace(
            base_cfg,
            authority_boost_enabled=False,
            recency_boost_enabled=False,
            profile_intent_boost_enabled=False,
        ),
    }

    base_probe_docs: dict[str, list[dict[str, Any]]] = {}
    per_query_rank_rows: list[dict[str, Any]] = []
    strict_rank_buckets: Counter[str] = Counter()
    support_rank_buckets: Counter[str] = Counter()
    strict_rr_values: list[float] = []
    support_rr_values: list[float] = []
    support_hit1_probe_count = 0
    support_hit3_probe_count = 0
    support_top1_equivalent_count = 0
    support_top3_queries = 0
    preferred_source_top1_when_support_top1 = 0
    preferred_source_in_top3_when_support_top3 = 0
    review_queue_rows: list[dict[str, Any]] = []
    review_reason_counter: Counter[str] = Counter()
    benchmark_mode_counter: Counter[str] = Counter()
    query_type_counter: Counter[str] = Counter()

    for row in rows:
        query_id = str(row.get("query_id"))
        query_text = str(row.get("query") or "")
        fixture_row = fixture_by_id.get(query_id, {})
        label_override = support_label_overrides.get(query_id)
        query_type = resolve_query_benchmark_type(
            fixture_query=fixture_row,
            label_override=label_override,
            default_query_type=default_positive_query_type,
        )
        benchmark_mode = benchmark_mode_for_query_type(query_type)
        query_type_counter[query_type] += 1
        benchmark_mode_counter[benchmark_mode] += 1

        strict_expected = _dedupe_str_checksums(row.get("expected_checksums") or [])
        support_expected, preferred_checksum = resolve_support_expectations(
            strict_expected_checksums=strict_expected,
            label_override=label_override,
        )

        retrieval = retrieve(query_text, cfg=base_cfg)
        docs = list(retrieval.documents)
        base_probe_docs[query_id] = docs

        for checksum in strict_expected:
            if isinstance(checksum, str) and checksum:
                cached_fulltext(checksum)
        for checksum in support_expected:
            if isinstance(checksum, str) and checksum:
                cached_fulltext(checksum)
        for doc in docs:
            checksum = doc.get("checksum")
            if isinstance(checksum, str) and checksum:
                cached_fulltext(checksum)

        strict_rank = _expected_rank(docs, strict_expected)
        preferred_rank = _expected_rank(docs, [preferred_checksum]) if preferred_checksum else None
        strict_rr = reciprocal_rank(strict_rank)
        strict_rr_values.append(strict_rr)
        strict_rank_buckets[rank_bucket(strict_rank, probe_depth)] += 1

        top1 = docs[0] if docs else None
        matched_expected = None
        if strict_rank is not None:
            matched_expected = docs[strict_rank - 1].get("checksum")

        first_support = find_first_answer_support(
            docs=docs,
            expected_checksums=support_expected,
            metadata_cache=metadata_cache,
            similarity_cache=similarity_cache,
            allow_equivalent=False,
        )
        support_rank = first_support.get("rank")
        support_rr = reciprocal_rank(support_rank if isinstance(support_rank, int) else None)
        support_rr_values.append(support_rr)
        support_rank_buckets[rank_bucket(support_rank if isinstance(support_rank, int) else None, probe_depth)] += 1
        support_hit_at_1 = support_rank == 1
        support_hit_at_3 = isinstance(support_rank, int) and support_rank <= 3
        if support_hit_at_1:
            support_hit1_probe_count += 1
            if first_support.get("support_type") == "equivalent":
                support_top1_equivalent_count += 1
            if preferred_checksum and first_support.get("checksum") == preferred_checksum:
                preferred_source_top1_when_support_top1 += 1
        if support_hit_at_3:
            support_hit3_probe_count += 1
            support_top3_queries += 1
            if preferred_rank is not None and preferred_rank <= 3:
                preferred_source_in_top3_when_support_top3 += 1

        archived_hit_at_1 = bool(row.get("hit_at_1"))
        archived_hit_at_3 = bool(row.get("hit_at_3"))
        failure_lens_hit_at_1 = (
            archived_hit_at_1
            if benchmark_mode == BENCHMARK_MODE_STRICT_RETRIEVAL
            else support_hit_at_1
        )
        failure_lens_hit_at_3 = (
            archived_hit_at_3
            if benchmark_mode == BENCHMARK_MODE_STRICT_RETRIEVAL
            else support_hit_at_3
        )
        selected_for_failure = not failure_lens_hit_at_1

        if selected_for_failure and benchmark_mode == BENCHMARK_MODE_ANSWER_SUPPORT:
            support_expected_set = set(support_expected)
            review_top_docs = []
            for review_rank, review_doc in enumerate(docs[: max(review_rank_limit, 1)], start=1):
                review_checksum = review_doc.get("checksum")
                review_top_docs.append(
                    {
                        "rank": review_rank,
                        "checksum": review_checksum,
                        "is_support_expected_checksum": (
                            isinstance(review_checksum, str)
                            and review_checksum in support_expected_set
                        ),
                        "features": _feature_snapshot(review_doc),
                    }
                )
            review_candidates = build_answer_support_review_candidates(
                query_text=query_text,
                docs=docs,
                support_expected_checksums=support_expected,
                metadata_cache=metadata_cache,
                similarity_cache=similarity_cache,
                rank_limit=max(review_rank_limit, 1),
            )
            for candidate in review_candidates:
                reason = candidate.get("review_reason")
                if isinstance(reason, str) and reason:
                    review_reason_counter[reason] += 1

            review_queue_rows.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "query_anchor_tokens": query_anchor_tokens(query_text),
                    "benchmark_query_type": query_type,
                    "benchmark_primary_mode": benchmark_mode,
                    "archived_hit_at_1": archived_hit_at_1,
                    "archived_hit_at_3": archived_hit_at_3,
                    "strict_expected_checksums_probe": strict_expected,
                    "answer_support_checksums_probe": support_expected,
                    "strict_retrieval_rank_probe": strict_rank,
                    "answer_support_rank_probe": support_rank,
                    "top_docs_probe": review_top_docs,
                    "suggested_candidate_docs": review_candidates,
                    "reviewer_selected_answer_support_checksums": [],
                    "reviewer_status": "pending",
                    "reviewer_notes": "",
                }
            )

        per_query_rank_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "benchmark_query_type": query_type,
                "benchmark_primary_mode": benchmark_mode,
                "selected_for_ranking_failure_analysis": selected_for_failure,
                "failure_lens_hit_at_1_probe": failure_lens_hit_at_1,
                "failure_lens_hit_at_3_probe": failure_lens_hit_at_3,
                "archived_hit_at_1": archived_hit_at_1,
                "archived_hit_at_3": archived_hit_at_3,
                "strict_expected_checksums_probe": strict_expected,
                "answer_support_checksums_probe": support_expected,
                "support_label_override_applied": bool(label_override),
                "preferred_expected_checksum": preferred_checksum,
                "preferred_expected_rank_probe": preferred_rank,
                "expected_rank_probe": strict_rank,
                "reciprocal_rank_probe": round(strict_rr, 4),
                "rank_bucket_probe": rank_bucket(strict_rank, probe_depth),
                "strict_retrieval_rank_probe": strict_rank,
                "strict_retrieval_hit_at_1_probe": strict_rank == 1,
                "strict_retrieval_hit_at_3_probe": isinstance(strict_rank, int)
                and strict_rank <= 3,
                "strict_retrieval_reciprocal_rank_probe": round(strict_rr, 4),
                "top1_checksum_probe": top1.get("checksum") if top1 else None,
                "top1_features_probe": _feature_snapshot(top1),
                "matched_expected_checksum_probe": matched_expected,
                "matched_expected_features_probe": _feature_snapshot(
                    docs[strict_rank - 1] if strict_rank is not None and strict_rank - 1 < len(docs) else None
                ),
                "answer_support_rank_probe": support_rank,
                "answer_support_hit_at_1_probe": support_hit_at_1,
                "answer_support_hit_at_3_probe": support_hit_at_3,
                "answer_support_reciprocal_rank_probe": round(support_rr, 4),
                "answer_support_first_doc_checksum_probe": first_support.get("checksum"),
                "answer_support_match_type_probe": first_support.get("support_type"),
                "answer_support_matched_expected_checksum_probe": first_support.get(
                    "matched_expected_checksum"
                ),
                "answer_support_similarity_probe": first_support.get("similarity"),
                "reviewer_answer_support_override": None,
                "reviewer_answer_support_notes": "",
            }
        )

    per_query_by_id = _query_rows_by_id(per_query_rank_rows)
    total_pos = len(rows)
    strict_hit1_probe = strict_rank_buckets.get(RANK_BUCKET_1, 0)
    strict_hit3_probe = strict_hit1_probe + strict_rank_buckets.get(RANK_BUCKET_2_3, 0)
    strict_mrr = sum(strict_rr_values) / len(strict_rr_values) if strict_rr_values else 0.0
    support_hit1_probe = support_rank_buckets.get(RANK_BUCKET_1, 0)
    support_hit3_probe = support_hit1_probe + support_rank_buckets.get(RANK_BUCKET_2_3, 0)
    support_mrr = sum(support_rr_values) / len(support_rr_values) if support_rr_values else 0.0
    strict_by_query_type = probe_metrics_by_query_type(
        per_query_rows=per_query_rank_rows,
        mode=BENCHMARK_MODE_STRICT_RETRIEVAL,
        probe_depth=probe_depth,
    )
    support_by_query_type = probe_metrics_by_query_type(
        per_query_rows=per_query_rank_rows,
        mode=BENCHMARK_MODE_ANSWER_SUPPORT,
        probe_depth=probe_depth,
    )
    query_type_counts = {
        query_type: query_type_counter.get(query_type, 0)
        for query_type in QUERY_TYPE_ORDER
    }
    primary_mode_counts = {
        BENCHMARK_MODE_STRICT_RETRIEVAL: benchmark_mode_counter.get(
            BENCHMARK_MODE_STRICT_RETRIEVAL, 0
        ),
        BENCHMARK_MODE_ANSWER_SUPPORT: benchmark_mode_counter.get(
            BENCHMARK_MODE_ANSWER_SUPPORT, 0
        ),
    }
    failed_query_ids = [
        str(row.get("query_id"))
        for row in per_query_rank_rows
        if bool(row.get("selected_for_ranking_failure_analysis"))
    ]
    failed_rows = [
        row_by_id[query_id] for query_id in failed_query_ids if query_id in row_by_id
    ]
    strict_canonical_failed_query_ids = [
        str(row.get("query_id"))
        for row in per_query_rank_rows
        if row.get("benchmark_query_type") == QUERY_TYPE_CANONICAL_DOCUMENT
        and not bool(row.get("strict_retrieval_hit_at_1_probe"))
    ]
    strict_canonical_rank_cause_rows: list[dict[str, Any]] = []
    strict_canonical_rank_cause_counter: Counter[str] = Counter()

    def _delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return round(a - b, 6)

    for query_id in strict_canonical_failed_query_ids:
        diag_row = per_query_by_id.get(query_id, {})
        query_text = str(diag_row.get("query") or "")
        docs = list(base_probe_docs.get(query_id, []))
        strict_expected = _dedupe_str_checksums(
            diag_row.get("strict_expected_checksums_probe") or []
        )
        preferred_expected_checksum = diag_row.get("preferred_expected_checksum")
        preferred_expected_checksum_str = (
            str(preferred_expected_checksum)
            if isinstance(preferred_expected_checksum, str) and preferred_expected_checksum
            else None
        )
        expected_rank_probe = _expected_rank(docs, strict_expected)
        expected_doc = (
            docs[expected_rank_probe - 1]
            if isinstance(expected_rank_probe, int)
            and expected_rank_probe > 0
            and expected_rank_probe - 1 < len(docs)
            else None
        )
        expected_checksum_for_diagnosis = (
            str(expected_doc.get("checksum"))
            if expected_doc and isinstance(expected_doc.get("checksum"), str)
            else (
                preferred_expected_checksum_str
                or (strict_expected[0] if strict_expected else None)
            )
        )
        if expected_doc is None and expected_checksum_for_diagnosis:
            expected_doc = _doc_by_checksum(docs, expected_checksum_for_diagnosis)

        winner_doc = docs[0] if docs else None
        winner_checksum = (
            str(winner_doc.get("checksum"))
            if winner_doc and isinstance(winner_doc.get("checksum"), str)
            else None
        )

        fusion_weight_vector, fusion_weight_bm25 = _fusion_weights_for_query(
            query_text, base_cfg
        )
        winner_contrib = _score_contributions(
            doc=winner_doc,
            fusion_weight_vector=fusion_weight_vector,
            fusion_weight_bm25=fusion_weight_bm25,
        )
        expected_contrib = _score_contributions(
            doc=expected_doc,
            fusion_weight_vector=fusion_weight_vector,
            fusion_weight_bm25=fusion_weight_bm25,
        )
        winner_vector = _safe_float(winner_contrib.get("vector_component"))
        expected_vector = _safe_float(expected_contrib.get("vector_component"))
        winner_lexical = _safe_float(winner_contrib.get("lexical_component"))
        expected_lexical = _safe_float(expected_contrib.get("lexical_component"))
        winner_prior = _safe_float(winner_contrib.get("doc_type_prior_component"))
        expected_prior = _safe_float(expected_contrib.get("doc_type_prior_component"))
        winner_final = _safe_float(winner_contrib.get("final_retrieval_score"))
        expected_final = _safe_float(expected_contrib.get("final_retrieval_score"))

        winner_title_overlap = _title_filename_overlap(
            query_text=query_text,
            doc=winner_doc,
        )
        expected_title_overlap = _title_filename_overlap(
            query_text=query_text,
            doc=expected_doc,
        )
        title_overlap_count_delta = (
            int(winner_title_overlap.get("overlap_count", 0))
            - int(expected_title_overlap.get("overlap_count", 0))
        )
        title_overlap_ratio_delta = round(
            float(winner_title_overlap.get("overlap_ratio", 0.0))
            - float(expected_title_overlap.get("overlap_ratio", 0.0)),
            4,
        )

        winner_chunk_signals = _chunk_aggregation_signals(winner_doc)
        expected_chunk_signals = _chunk_aggregation_signals(expected_doc)
        winner_has_dual_signal = bool(winner_chunk_signals.get("has_vector_signal")) and bool(
            winner_chunk_signals.get("has_lexical_signal")
        )
        expected_has_dual_signal = bool(
            expected_chunk_signals.get("has_vector_signal")
        ) and bool(expected_chunk_signals.get("has_lexical_signal"))
        winner_cv_suppressed = _safe_float(winner_chunk_signals.get("cv_family_suppressed")) or 0.0
        chunk_aggregation_bias = bool(
            isinstance(expected_rank_probe, int)
            and expected_rank_probe <= 10
            and winner_has_dual_signal
            and (not expected_has_dual_signal or winner_cv_suppressed > 0.0)
            and (
                _delta(winner_final, expected_final) is None
                or (_delta(winner_final, expected_final) or 0.0) <= 0.20
            )
        )

        winner_meta = cached_fulltext(winner_checksum) if winner_checksum else None
        expected_meta = (
            cached_fulltext(expected_checksum_for_diagnosis)
            if expected_checksum_for_diagnosis
            else None
        )
        winner_text = str((winner_meta or {}).get("text_full") or "")
        expected_text = str((expected_meta or {}).get("text_full") or "")
        winner_expected_similarity = (
            text_similarity_metrics(winner_text, expected_text)
            if winner_text and expected_text
            else None
        )
        near_duplicate_collision = bool(
            winner_expected_similarity
            and (
                bool(winner_expected_similarity.get("near_duplicate"))
                or (
                    float(winner_expected_similarity.get("containment_min", 0.0) or 0.0)
                    >= 0.88
                    and float(winner_expected_similarity.get("sequence_ratio", 0.0) or 0.0)
                    >= 0.75
                )
            )
        )

        cause_bucket = assign_primary_ranking_cause(
            expected_rank_probe=expected_rank_probe,
            winner_vector_minus_expected=_delta(winner_vector, expected_vector),
            winner_lexical_minus_expected=_delta(winner_lexical, expected_lexical),
            title_overlap_count_delta=title_overlap_count_delta,
            title_overlap_ratio_delta=title_overlap_ratio_delta,
            doc_type_prior_delta=_delta(winner_prior, expected_prior),
            near_duplicate_collision=near_duplicate_collision,
            chunk_aggregation_bias=chunk_aggregation_bias,
        )
        strict_canonical_rank_cause_counter[cause_bucket] += 1

        strict_canonical_rank_cause_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "expected_checksum": expected_checksum_for_diagnosis,
                "actual_top1_checksum": winner_checksum,
                "expected_rank_if_retrieved": expected_rank_probe,
                "winner_doc_metadata": {
                    "path": (winner_meta or {}).get("path"),
                    "filename": (winner_meta or {}).get("filename"),
                    "doc_type": (winner_meta or {}).get("doc_type"),
                },
                "expected_doc_metadata": {
                    "path": (expected_meta or {}).get("path"),
                    "filename": (expected_meta or {}).get("filename"),
                    "doc_type": (expected_meta or {}).get("doc_type"),
                },
                "winner_vs_expected_diagnostics": {
                    "lexical_score_contribution": {
                        "winner": winner_lexical,
                        "expected": expected_lexical,
                        "delta_winner_minus_expected": _delta(
                            winner_lexical, expected_lexical
                        ),
                    },
                    "vector_score_contribution": {
                        "winner": winner_vector,
                        "expected": expected_vector,
                        "delta_winner_minus_expected": _delta(
                            winner_vector, expected_vector
                        ),
                    },
                    "title_filename_overlap_features": {
                        "winner": winner_title_overlap,
                        "expected": expected_title_overlap,
                        "delta_overlap_count_winner_minus_expected": title_overlap_count_delta,
                        "delta_overlap_ratio_winner_minus_expected": title_overlap_ratio_delta,
                    },
                    "doc_type_prior_contribution": {
                        "winner": winner_prior,
                        "expected": expected_prior,
                        "delta_winner_minus_expected": _delta(
                            winner_prior, expected_prior
                        ),
                    },
                    "chunk_aggregation_signals": {
                        "winner": winner_chunk_signals,
                        "expected": expected_chunk_signals,
                        "heuristic_chunk_aggregation_bias": chunk_aggregation_bias,
                    },
                    "final_score_delta": {
                        "winner_retrieval_score": winner_final,
                        "expected_retrieval_score": expected_final,
                        "delta_winner_minus_expected": _delta(
                            winner_final, expected_final
                        ),
                    },
                    "winner_expected_text_similarity": winner_expected_similarity,
                },
                "primary_ranking_cause_bucket": cause_bucket,
                "reviewer_bucket_override": None,
                "reviewer_notes": "",
            }
        )
    strict_canonical_bucket_counts = {
        bucket: int(strict_canonical_rank_cause_counter.get(bucket, 0))
        for bucket in RANKING_CAUSE_ORDER
    }
    strict_canonical_dominant_bucket = max(
        RANKING_CAUSE_ORDER,
        key=lambda bucket: strict_canonical_rank_cause_counter.get(bucket, 0),
    )
    strict_canonical_hard_negative_analysis = analyze_strict_canonical_hard_negatives(
        strict_canonical_rank_cause_rows
    )
    strict_canonical_cleaned_residual_split = (
        build_strict_canonical_cleaned_residual_split(
            archived_rows=rows,
            per_query_rows=per_query_rank_rows,
            strict_canonical_rows=strict_canonical_rank_cause_rows,
            strict_canonical_hard_negative_analysis=strict_canonical_hard_negative_analysis,
            metadata_lookup=cached_fulltext,
            artifact_near_tie_epsilon=artifact_near_tie_epsilon,
        )
    )
    ablation_per_query: list[dict[str, Any]] = []
    ablation_deltas: dict[str, list[int]] = defaultdict(list)

    for query_id in failed_query_ids:
        diag_row = per_query_by_id.get(query_id, {})
        benchmark_mode = str(
            diag_row.get("benchmark_primary_mode") or BENCHMARK_MODE_STRICT_RETRIEVAL
        )
        query_type = str(diag_row.get("benchmark_query_type") or QUERY_TYPE_CANONICAL_DOCUMENT)
        row = row_by_id[query_id]
        query_text = str(row.get("query") or "")
        strict_expected = _dedupe_str_checksums(
            diag_row.get("strict_expected_checksums_probe") or row.get("expected_checksums") or []
        )
        support_expected = _dedupe_str_checksums(
            diag_row.get("answer_support_checksums_probe") or strict_expected
        )
        expected_for_failure_lens = (
            support_expected
            if benchmark_mode == BENCHMARK_MODE_ANSWER_SUPPORT
            else strict_expected
        )
        base_rank = _expected_rank(base_probe_docs.get(query_id, []), expected_for_failure_lens)
        ablation_result = {
            "query_id": query_id,
            "query": query_text,
            "benchmark_query_type": query_type,
            "benchmark_primary_mode": benchmark_mode,
            "expected_checksums_for_failure_lens": expected_for_failure_lens,
            "ranks": {},
            "rank_delta_vs_base": {},
        }
        base_rank_value = _rank_value_for_delta(base_rank, probe_depth)
        ablation_result["ranks"]["base"] = base_rank

        for name, cfg in ablation_cfgs.items():
            if name == "base":
                continue
            docs = list(retrieve(query_text, cfg=cfg).documents)
            rank = _expected_rank(docs, expected_for_failure_lens)
            ablation_result["ranks"][name] = rank
            delta = _rank_value_for_delta(rank, probe_depth) - base_rank_value
            ablation_result["rank_delta_vs_base"][name] = delta
            ablation_deltas[name].append(delta)

        ablation_per_query.append(ablation_result)

    ablation_summary = []
    for name in ("no_authority", "no_recency", "no_profile_intent", "no_boosts"):
        deltas = ablation_deltas.get(name, [])
        improved = sum(1 for d in deltas if d < 0)
        worsened = sum(1 for d in deltas if d > 0)
        unchanged = sum(1 for d in deltas if d == 0)
        avg_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
        ablation_summary.append(
            {
                "ablation": name,
                "queries": len(deltas),
                "improved": improved,
                "worsened": worsened,
                "unchanged": unchanged,
                "avg_rank_delta_vs_base": round(avg_delta, 3),
            }
        )

    similarity_rows: list[dict[str, Any]] = []
    similarity_counter: Counter[str] = Counter()
    likely_correct_count = 0

    for row in failed_rows:
        query_id = str(row.get("query_id") or "")
        per_query_diag = per_query_by_id.get(query_id, {})
        strict_expected = _dedupe_str_checksums(
            per_query_diag.get("strict_expected_checksums_probe")
            or row.get("expected_checksums")
            or []
        )
        support_expected = _dedupe_str_checksums(
            per_query_diag.get("answer_support_checksums_probe") or strict_expected
        )
        benchmark_mode = str(
            per_query_diag.get("benchmark_primary_mode") or BENCHMARK_MODE_STRICT_RETRIEVAL
        )
        query_type = str(
            per_query_diag.get("benchmark_query_type") or QUERY_TYPE_CANONICAL_DOCUMENT
        )
        top1_checksum = row.get("top1_checksum")
        if not isinstance(top1_checksum, str) or not top1_checksum:
            continue

        top1_meta = cached_fulltext(top1_checksum) or {}
        top1_text = str(top1_meta.get("text_full") or "")

        expected_text_by_checksum: dict[str, str] = {}
        best_expected_meta: dict[str, Any] | None = None
        for checksum in support_expected:
            expected_meta = cached_fulltext(checksum)
            if not expected_meta:
                continue
            expected_text = str(expected_meta.get("text_full") or "")
            if expected_text:
                expected_text_by_checksum[checksum] = expected_text

        best_expected_checksum, best_similarity = _best_similarity_to_expected(
            doc_checksum=top1_checksum,
            doc_text=top1_text,
            expected_checksums=support_expected,
            expected_text_by_checksum=expected_text_by_checksum,
            similarity_cache=similarity_cache,
        )
        if best_expected_checksum:
            best_expected_meta = cached_fulltext(best_expected_checksum) or {}

        near_duplicate = bool(best_similarity.get("near_duplicate")) if best_similarity else False
        containment = float(best_similarity.get("containment_min", 0.0)) if best_similarity else 0.0
        strict_rank_probe = _expected_rank(base_probe_docs.get(query_id, []), strict_expected)
        support_rank_probe_computed = _expected_rank(
            base_probe_docs.get(query_id, []), support_expected
        )
        support_rank_probe = per_query_diag.get("answer_support_rank_probe")
        if support_rank_probe is None:
            support_rank_probe = support_rank_probe_computed
        strict_hit3_probe_row = bool(
            isinstance(strict_rank_probe, int) and strict_rank_probe <= 3
        )
        if bool(per_query_diag.get("strict_retrieval_hit_at_3_probe")):
            strict_hit3_probe_row = True
        support_hit3_probe_row = bool(
            isinstance(support_rank_probe, int) and support_rank_probe <= 3
        )
        if bool(per_query_diag.get("answer_support_hit_at_3_probe")):
            support_hit3_probe_row = True
        hit3_for_likelihood = (
            strict_hit3_probe_row
            if benchmark_mode == BENCHMARK_MODE_STRICT_RETRIEVAL
            else support_hit3_probe_row
        )
        rank_for_likelihood = (
            strict_rank_probe
            if benchmark_mode == BENCHMARK_MODE_STRICT_RETRIEVAL
            else support_rank_probe
        )

        likelihood = auto_answer_likelihood(
            hit_at_3=hit3_for_likelihood,
            expected_rank=rank_for_likelihood if isinstance(rank_for_likelihood, int) else None,
            near_duplicate=near_duplicate,
            similarity_containment=containment,
        )
        similarity_counter[likelihood] += 1
        if likelihood == LIKELY_CORRECT:
            likely_correct_count += 1

        similarity_rows.append(
            {
                "query_id": query_id,
                "query": row.get("query"),
                "benchmark_query_type": query_type,
                "benchmark_primary_mode": benchmark_mode,
                "top1_checksum": top1_checksum,
                "top1_path": top1_meta.get("path"),
                "top1_doc_type": top1_meta.get("doc_type"),
                "strict_expected_checksums_probe": strict_expected,
                "answer_support_checksums_probe": support_expected,
                "best_expected_checksum": best_expected_checksum,
                "best_expected_path": (best_expected_meta or {}).get("path"),
                "best_expected_doc_type": (best_expected_meta or {}).get("doc_type"),
                "similarity": best_similarity,
                "strict_retrieval_rank_probe": strict_rank_probe,
                "answer_support_rank_probe": support_rank_probe,
                "answer_support_match_type_probe": per_query_diag.get(
                    "answer_support_match_type_probe"
                ),
                "hit_at_3_archived": bool(row.get("hit_at_3")),
                "strict_retrieval_hit_at_3_probe": strict_hit3_probe_row,
                "answer_support_hit_at_3_probe": support_hit3_probe_row,
                "failure_lens_hit_at_3_probe": hit3_for_likelihood,
                "auto_answer_likelihood": likelihood,
                "reviewer_answer_correct": None,
                "reviewer_notes": "",
            }
        )

    hard_negatives = _most_common_wrong_top1(failed_rows, metadata_cache)
    review_queries_total = len(review_queue_rows)
    review_queries_with_candidates = sum(
        1 for row in review_queue_rows if row.get("suggested_candidate_docs")
    )
    review_candidates_total = sum(
        len(row.get("suggested_candidate_docs") or []) for row in review_queue_rows
    )
    probe_vs_eval_comparison = build_probe_vs_eval_comparison(
        patha_runbook_path=patha_runbook_path,
        ranking_artifact_path=output_path,
        runbook=runbook,
        archived_rows=rows,
        per_query_rows=per_query_rank_rows,
        probe_docs_by_query=base_probe_docs,
        probe_cfg=base_cfg,
    )

    output = {
        "schema_version": RANKING_INVESTIGATION_SCHEMA_VERSION,
        "compatibility_note": (
            "Schema v5 keeps v4 outputs and adds strict_canonical_cleaned_residual_split "
            "for benchmark-cleaned strict canonical triage "
            "(actionable vs ambiguity vs already-addressed artifact-first)."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "patha_runbook": str(patha_runbook_path),
            "fixture": str(fixture_path),
            "support_labels": str(support_labels_path) if support_labels_path else None,
        },
        "methodology": {
            "deterministic_probe": True,
            "note": (
                "Deep ranking probe runs retrieval with variants disabled (exact query only) "
                "to avoid rewrite-model variance while investigating ranking behavior."
            ),
            "benchmark_modes": [
                BENCHMARK_MODE_STRICT_RETRIEVAL,
                BENCHMARK_MODE_ANSWER_SUPPORT,
            ],
            "query_type_labels": list(QUERY_TYPE_ORDER),
            "default_positive_query_type": default_positive_query_type,
            "support_labels_overrides_count": len(support_label_overrides),
            "answer_support_policy": {
                "strict_support": "retrieved checksum in expected_checksums",
                "equivalent_support_scoring": (
                    "disabled for benchmark scoring in this step; only manual "
                    "answer_support_checksums labels are counted"
                ),
                "equivalent_thresholds": {
                    "containment_min": EQUIV_SUPPORT_CONTAINMENT_MIN,
                    "sequence_ratio_min": EQUIV_SUPPORT_SEQUENCE_MIN,
                    "alt_containment_min": EQUIV_SUPPORT_ALT_CONTAINMENT_MIN,
                    "alt_jaccard_min": EQUIV_SUPPORT_ALT_JACCARD_MIN,
                    "alt_sequence_ratio_min": EQUIV_SUPPORT_ALT_SEQUENCE_MIN,
                },
                "review_candidate_policy": {
                    "query_conditioned_containment_min": REVIEW_QUERY_CONTAINMENT_MIN,
                    "query_conditioned_sequence_ratio_min": REVIEW_QUERY_SEQUENCE_MIN,
                    "query_conditioned_jaccard_min": REVIEW_QUERY_JACCARD_MIN,
                    "query_anchor_coverage_min": REVIEW_ANCHOR_COVERAGE_MIN,
                    "review_rank_limit": max(review_rank_limit, 1),
                },
            },
            "probe_depth": probe_depth,
            "probe_config": {
                "top_k": base_cfg.top_k,
                "top_k_each": base_cfg.top_k_each,
                "enable_variants": base_cfg.enable_variants,
                "enable_mmr": base_cfg.enable_mmr,
                "anchored_fusion_weight_vector": base_cfg.anchored_fusion_weight_vector,
                "anchored_fusion_weight_bm25": base_cfg.anchored_fusion_weight_bm25,
            },
        },
        "archived_patha_metrics": runbook.get("summary", {}),
        "probe_vs_eval_comparison": probe_vs_eval_comparison,
        "benchmark_scope_summary": {
            "query_type_counts": query_type_counts,
            "primary_mode_counts": primary_mode_counts,
            "ranking_failure_query_count": len(failed_query_ids),
        },
        "probe_ranking_metrics": {
            BENCHMARK_MODE_STRICT_RETRIEVAL: {
                "positive_total": total_pos,
                "hit_at_1_probe": strict_hit1_probe,
                "hit_at_1_probe_rate": round((strict_hit1_probe / total_pos), 4)
                if total_pos
                else 0.0,
                "hit_at_3_probe": strict_hit3_probe,
                "hit_at_3_probe_rate": round((strict_hit3_probe / total_pos), 4)
                if total_pos
                else 0.0,
                "mrr_probe": round(strict_mrr, 4),
                "rank_bucket_counts": {
                    RANK_BUCKET_1: strict_rank_buckets.get(RANK_BUCKET_1, 0),
                    RANK_BUCKET_2_3: strict_rank_buckets.get(RANK_BUCKET_2_3, 0),
                    RANK_BUCKET_4_10: strict_rank_buckets.get(RANK_BUCKET_4_10, 0),
                    RANK_BUCKET_11_TO_DEPTH: strict_rank_buckets.get(RANK_BUCKET_11_TO_DEPTH, 0),
                    RANK_BUCKET_NOT_RETRIEVED: strict_rank_buckets.get(
                        RANK_BUCKET_NOT_RETRIEVED, 0
                    ),
                },
                "by_query_type": strict_by_query_type,
            },
            BENCHMARK_MODE_ANSWER_SUPPORT: {
                "positive_total": total_pos,
                "hit_at_1_probe": support_hit1_probe,
                "hit_at_1_probe_rate": round((support_hit1_probe / total_pos), 4)
                if total_pos
                else 0.0,
                "hit_at_3_probe": support_hit3_probe,
                "hit_at_3_probe_rate": round((support_hit3_probe / total_pos), 4)
                if total_pos
                else 0.0,
                "mrr_probe": round(support_mrr, 4),
                "rank_bucket_counts": {
                    RANK_BUCKET_1: support_rank_buckets.get(RANK_BUCKET_1, 0),
                    RANK_BUCKET_2_3: support_rank_buckets.get(RANK_BUCKET_2_3, 0),
                    RANK_BUCKET_4_10: support_rank_buckets.get(RANK_BUCKET_4_10, 0),
                    RANK_BUCKET_11_TO_DEPTH: support_rank_buckets.get(RANK_BUCKET_11_TO_DEPTH, 0),
                    RANK_BUCKET_NOT_RETRIEVED: support_rank_buckets.get(
                        RANK_BUCKET_NOT_RETRIEVED, 0
                    ),
                },
                "by_query_type": support_by_query_type,
                "attribution_quality": {
                    "support_top1_queries": support_hit1_probe_count,
                    "preferred_source_top1_when_support_top1": preferred_source_top1_when_support_top1,
                    "preferred_source_top1_rate_when_support_top1": round(
                        (preferred_source_top1_when_support_top1 / support_hit1_probe_count), 4
                    )
                    if support_hit1_probe_count
                    else 0.0,
                    "support_top1_equivalent_source_count": support_top1_equivalent_count,
                    "support_top3_queries": support_top3_queries,
                    "preferred_source_in_top3_when_support_top3": preferred_source_in_top3_when_support_top3,
                    "preferred_source_in_top3_rate_when_support_top3": round(
                        (preferred_source_in_top3_when_support_top3 / support_top3_queries), 4
                    )
                    if support_top3_queries
                    else 0.0,
                },
            },
            "comparison": {
                "answer_support_lift_vs_strict_at_1": support_hit1_probe - strict_hit1_probe,
                "answer_support_lift_vs_strict_at_3": support_hit3_probe - strict_hit3_probe,
            },
        },
        "per_query_rank_diagnostics": per_query_rank_rows,
        "strict_canonical_ranking_diagnosis": {
            "scope": (
                "strict canonical ranking misses only "
                "(benchmark_query_type=canonical_document_query and strict hit@1 miss)"
            ),
            "failed_query_count": len(strict_canonical_rank_cause_rows),
            "bucket_counts": strict_canonical_bucket_counts,
            "largest_bucket": {
                "bucket": strict_canonical_dominant_bucket,
                "count": int(
                    strict_canonical_rank_cause_counter.get(
                        strict_canonical_dominant_bucket, 0
                    )
                ),
                "share_of_failed_queries": round(
                    (
                        strict_canonical_rank_cause_counter.get(
                            strict_canonical_dominant_bucket, 0
                        )
                        / len(strict_canonical_rank_cause_rows)
                    ),
                    4,
                )
                if strict_canonical_rank_cause_rows
                else 0.0,
            },
            "rows": strict_canonical_rank_cause_rows,
            "hard_negative_analysis": strict_canonical_hard_negative_analysis,
        },
        "strict_canonical_cleaned_residual_split": strict_canonical_cleaned_residual_split,
        "ablation_analysis": {
            "scope": (
                "benchmark-aware ranking failures: strict_retrieval misses for canonical/ambiguous "
                "queries, answer_support misses for explicitly multi-source queries"
            ),
            "query_count": len(failed_query_ids),
            "summary": ablation_summary,
            "per_query": ablation_per_query,
        },
        "content_similarity_analysis": {
            "scope": "benchmark-aware ranking failures",
            "summary": {
                "queries_analyzed": len(similarity_rows),
                "auto_answer_likelihood_counts": dict(similarity_counter),
                "likely_correct_despite_rank_miss": likely_correct_count,
            },
            "failed_query_rows": similarity_rows,
            "repeated_wrong_top1_hard_negatives": hard_negatives,
        },
        "answer_support_review_queue": {
            "scope": "benchmark_primary_mode=answer_support and benchmark-selected failures",
            "summary": {
                "queries_in_queue": review_queries_total,
                "queries_with_suggested_candidates": review_queries_with_candidates,
                "queries_with_suggested_candidates_rate": round(
                    (review_queries_with_candidates / review_queries_total), 4
                )
                if review_queries_total
                else 0.0,
                "suggested_candidate_docs_total": review_candidates_total,
                "suggested_review_reason_counts": dict(review_reason_counter),
                "instruction": (
                    "Review suggested candidates and fill "
                    "reviewer_selected_answer_support_checksums for queries where "
                    "alternate sources should count as answer-support."
                ),
            },
            "rows": review_queue_rows,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    if cleaned_strict_output_path is not None:
        cleaned_sidecar = {
            "schema_version": STRICT_CANONICAL_CLEANED_RESIDUAL_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_artifacts": {
                "patha_runbook": str(patha_runbook_path),
                "ranking_investigation": str(output_path),
            },
            "strict_canonical_cleaned_residual_split": strict_canonical_cleaned_residual_split,
        }
        cleaned_strict_output_path.parent.mkdir(parents=True, exist_ok=True)
        with cleaned_strict_output_path.open("w", encoding="utf-8") as fh:
            json.dump(cleaned_sidecar, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    return output


def _default_output(patha_runbook: Path) -> Path:
    return patha_runbook.with_name(f"{patha_runbook.stem}_ranking_investigation.json")


def _default_cleaned_strict_output(ranking_output: Path) -> Path:
    return ranking_output.with_name(
        f"{ranking_output.stem}_strict_canonical_cleaned_residuals.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Investigate residual ranking weakness after Path A without changing production behavior."
    )
    parser.add_argument(
        "--patha-runbook",
        type=Path,
        default=Path("docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1.json"),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/retrieval_eval_queries.json"),
    )
    parser.add_argument(
        "--support-labels",
        type=Path,
        default=DEFAULT_SUPPORT_LABELS_PATH,
        help=(
            "Optional JSON file containing per-query answer-support overrides "
            "(manual support checksums, query benchmark type, preferred checksum)."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cleaned-strict-output",
        type=Path,
        default=None,
        help=(
            "Optional sidecar output path for benchmark-cleaned strict canonical split "
            "(actionable vs ambiguity vs already-addressed artifact-first)."
        ),
    )
    parser.add_argument("--probe-depth", type=int, default=DEFAULT_PROBE_DEPTH)
    parser.add_argument(
        "--artifact-near-tie-epsilon",
        type=float,
        default=DEFAULT_ARTIFACT_NEAR_TIE_EPSILON,
        help=(
            "Score delta threshold used to flag already-addressed artifact-first cases "
            "where artifact docs remain near-tied behind a correct top-1."
        ),
    )
    parser.add_argument(
        "--review-rank-limit",
        type=int,
        default=DEFAULT_REVIEW_RANK_LIMIT,
        help="Top-N probe docs to include per failed query in answer-support review queue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or _default_output(args.patha_runbook)
    cleaned_strict_output_path = (
        args.cleaned_strict_output or _default_cleaned_strict_output(output_path)
    )
    output = investigate_ranking_post_patha(
        patha_runbook_path=args.patha_runbook,
        fixture_path=args.fixture,
        output_path=output_path,
        probe_depth=max(args.probe_depth, 1),
        support_labels_path=args.support_labels,
        review_rank_limit=max(args.review_rank_limit, 1),
        cleaned_strict_output_path=cleaned_strict_output_path,
        artifact_near_tie_epsilon=max(float(args.artifact_near_tie_epsilon), 0.0),
    )
    metrics = output.get("probe_ranking_metrics", {})
    strict_metrics = metrics.get(BENCHMARK_MODE_STRICT_RETRIEVAL, {})
    support_metrics = metrics.get(BENCHMARK_MODE_ANSWER_SUPPORT, {})
    print(f"Wrote: {output_path}")
    print(f"Wrote cleaned strict sidecar: {cleaned_strict_output_path}")
    print(
        "Probe metrics: "
        f"strict_hit@1={strict_metrics.get('hit_at_1_probe')}/{strict_metrics.get('positive_total')}, "
        f"strict_hit@3={strict_metrics.get('hit_at_3_probe')}/{strict_metrics.get('positive_total')}, "
        f"support_hit@3={support_metrics.get('hit_at_3_probe')}/{support_metrics.get('positive_total')}, "
        f"strict_mrr={strict_metrics.get('mrr_probe')}, "
        f"support_mrr={support_metrics.get('mrr_probe')}"
    )


if __name__ == "__main__":
    main()
