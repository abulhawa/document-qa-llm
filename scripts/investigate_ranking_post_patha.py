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
RANKING_INVESTIGATION_SCHEMA_VERSION = "ranking_investigation.v2"
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

    output = {
        "schema_version": RANKING_INVESTIGATION_SCHEMA_VERSION,
        "compatibility_note": (
            "Schema v2 uses nested probe_ranking_metrics.{strict_retrieval,answer_support} "
            "blocks and includes by_query_type metrics. Legacy flat probe keys are not emitted."
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

    return output


def _default_output(patha_runbook: Path) -> Path:
    return patha_runbook.with_name(f"{patha_runbook.stem}_ranking_investigation.json")


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
    parser.add_argument("--probe-depth", type=int, default=DEFAULT_PROBE_DEPTH)
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
    output = investigate_ranking_post_patha(
        patha_runbook_path=args.patha_runbook,
        fixture_path=args.fixture,
        output_path=output_path,
        probe_depth=max(args.probe_depth, 1),
        support_labels_path=args.support_labels,
        review_rank_limit=max(args.review_rank_limit, 1),
    )
    metrics = output.get("probe_ranking_metrics", {})
    strict_metrics = metrics.get(BENCHMARK_MODE_STRICT_RETRIEVAL, {})
    support_metrics = metrics.get(BENCHMARK_MODE_ANSWER_SUPPORT, {})
    print(f"Wrote: {output_path}")
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
