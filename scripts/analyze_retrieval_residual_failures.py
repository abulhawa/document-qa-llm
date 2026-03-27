#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Ensure repo root is on sys.path when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CHUNKS_INDEX, FULLTEXT_INDEX
from core.opensearch_client import get_client
from core.query_rewriter import has_strong_query_anchors
from core.retrieval.pipeline import retrieve
from core.retrieval.types import RetrievalConfig
from utils.qdrant_utils import count_qdrant_chunks_by_checksum


BUCKET_TOP3_NOT_TOP1 = "relevant doc ranked below 1 but within top-3"
BUCKET_RETRIEVED_BELOW_TOP3 = "relevant doc retrieved but ranked below top-3"
BUCKET_NOT_RETRIEVED_TEXT_AVAILABLE = "relevant doc not retrieved despite text being available"
BUCKET_TEXT_GAP = "likely text extraction / OCR gap"
BUCKET_CORPUS_ABSENCE = "likely corpus absence or mislabeled expectation"
BUCKET_AMBIGUOUS = "ambiguous / needs manual review"

BUCKET_ORDER: tuple[str, ...] = (
    BUCKET_TOP3_NOT_TOP1,
    BUCKET_RETRIEVED_BELOW_TOP3,
    BUCKET_NOT_RETRIEVED_TEXT_AVAILABLE,
    BUCKET_TEXT_GAP,
    BUCKET_CORPUS_ABSENCE,
    BUCKET_AMBIGUOUS,
)

_ANCHOR_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")
_SEMI_ANCHOR_TERMS = {
    "ali",
    "cv",
    "resume",
    "cover",
    "letter",
    "report",
    "paper",
    "course",
    "lecture",
    "project",
    "insurance",
    "invoice",
    "tax",
    "receipt",
    "jobcenter",
    "formular",
    "form",
}
_OCR_EXTRACTION_MODES = {"ocr", "ocr_local", "ocr_groq"}
_OCR_MARKER_FIELDS = (
    "ocr_confidence_mean",
    "ocr_confidence_min",
    "ocr_page_count",
    "ocr_fallback_used",
)

DEFAULT_CANDIDATE_DEPTH = 60
DEFAULT_VERY_LOW_TEXT_THRESHOLD = 200
OCR_MEANINGFUL_MIN_RATE = 0.30
OCR_MEANINGFUL_MIN_COUNT = 3
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
RESIDUAL_FAILURE_ANALYSIS_SCHEMA_VERSION = "residual_failure_analysis.v2"


@dataclass(frozen=True)
class ExpectedDocIndicator:
    checksum: str
    exists_in_fulltext: bool
    path: str | None
    filename: str | None
    doc_type: str | None
    extraction_mode: str | None
    text_length: int | None
    empty_text: bool | None
    very_low_text_length: bool | None
    has_ocr_marker: bool | None
    os_chunk_count: int | None
    qdrant_chunk_count: int | None
    missing_chunk_coverage: bool | None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


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


def _tokenize(query: str) -> list[str]:
    return [tok.lower() for tok in _ANCHOR_TOKEN_RE.findall(query or "")]


def classify_query_anchor(query: str) -> str:
    if has_strong_query_anchors(query):
        return "anchored"

    lowered = _tokenize(query)
    if any(tok in _SEMI_ANCHOR_TERMS for tok in lowered):
        return "semi-anchored"
    if any(tok.isdigit() and len(tok) >= 4 for tok in lowered):
        return "semi-anchored"
    if re.search(r"\b[a-z]{2,}'s\b", query or "", flags=re.IGNORECASE):
        return "semi-anchored"
    if re.search(r"\b[a-z0-9]+-[a-z0-9]+\b", query or "", flags=re.IGNORECASE):
        return "semi-anchored"
    return "unanchored"


def assign_primary_bucket(
    *,
    relevant_in_top3: bool,
    relevant_in_candidates: bool | None,
    expected_rank_in_candidates: int | None,
    text_gap_evidence: bool,
    corpus_absence_evidence: bool,
    text_available_evidence: bool,
) -> str:
    if relevant_in_top3:
        return BUCKET_TOP3_NOT_TOP1
    if relevant_in_candidates:
        if expected_rank_in_candidates is None:
            return BUCKET_AMBIGUOUS
        if expected_rank_in_candidates <= 3:
            return BUCKET_AMBIGUOUS
        return BUCKET_RETRIEVED_BELOW_TOP3
    if corpus_absence_evidence:
        return BUCKET_CORPUS_ABSENCE
    if text_gap_evidence:
        return BUCKET_TEXT_GAP
    if relevant_in_candidates is False and text_available_evidence:
        return BUCKET_NOT_RETRIEVED_TEXT_AVAILABLE
    return BUCKET_AMBIGUOUS


def build_ocr_recommendation(
    bucket_counts: Mapping[str, int],
    total_failed: int,
) -> dict[str, Any]:
    text_gap_count = int(bucket_counts.get(BUCKET_TEXT_GAP, 0))
    text_gap_rate = (text_gap_count / total_failed) if total_failed else 0.0
    yes = (
        total_failed > 0
        and text_gap_count >= OCR_MEANINGFUL_MIN_COUNT
        and text_gap_rate >= OCR_MEANINGFUL_MIN_RATE
    )

    dominant_bucket = BUCKET_AMBIGUOUS
    dominant_count = 0
    for bucket in BUCKET_ORDER:
        count = int(bucket_counts.get(bucket, 0))
        if count > dominant_count:
            dominant_bucket = bucket
            dominant_count = count

    if yes:
        rationale = (
            "YES: a meaningful share of residual misses shows text-extraction gap evidence."
        )
    else:
        rationale = (
            "NO: residual misses are dominated by non-OCR causes; prioritize ranking/retrieval fixes first."
        )

    return {
        "decision": "YES" if yes else "NO",
        "recommend_ocr_canary_now": yes,
        "criteria": {
            "min_text_gap_rate": OCR_MEANINGFUL_MIN_RATE,
            "min_text_gap_count": OCR_MEANINGFUL_MIN_COUNT,
        },
        "evidence": {
            "text_gap_failures": text_gap_count,
            "total_failed_positives": total_failed,
            "text_gap_rate": round(text_gap_rate, 4),
            "dominant_bucket": dominant_bucket,
            "dominant_bucket_count": dominant_count,
        },
        "rationale": rationale,
    }


def _build_probe_config(
    patha_config: Mapping[str, Any],
    *,
    candidate_depth: int,
) -> RetrievalConfig:
    top_k = max(candidate_depth, 1)
    top_k_each = max(top_k * 4, 40)
    return RetrievalConfig(
        top_k=top_k,
        top_k_each=top_k_each,
        enable_variants=False,
        enable_mmr=bool(patha_config.get("enable_mmr", True)),
        anchored_exact_only=bool(patha_config.get("anchored_exact_only", True)),
        anchored_lexical_bias_enabled=bool(
            patha_config.get("anchored_lexical_bias_enabled", True)
        ),
        anchored_fusion_weight_vector=float(
            patha_config.get("anchored_fusion_weight_vector", 0.4)
        ),
        anchored_fusion_weight_bm25=float(
            patha_config.get("anchored_fusion_weight_bm25", 0.6)
        ),
        fusion_weight_vector=float(patha_config.get("fusion_weight_vector", 0.7)),
        fusion_weight_bm25=float(patha_config.get("fusion_weight_bm25", 0.3)),
    )


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


def _count_os_chunks_by_checksum(os_client: Any, checksum: str) -> int | None:
    try:
        response = os_client.count(
            index=CHUNKS_INDEX,
            body={"query": {"term": {"checksum": {"value": checksum}}}},
        )
    except Exception:
        return None
    value = response.get("count")
    if isinstance(value, int):
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _indicator_from_fulltext(
    *,
    checksum: str,
    fulltext_doc: dict[str, Any] | None,
    os_chunk_count: int | None,
    qdrant_chunk_count: int | None,
    very_low_text_threshold: int,
) -> ExpectedDocIndicator:
    if not fulltext_doc:
        return ExpectedDocIndicator(
            checksum=checksum,
            exists_in_fulltext=False,
            path=None,
            filename=None,
            doc_type=None,
            extraction_mode=None,
            text_length=None,
            empty_text=None,
            very_low_text_length=None,
            has_ocr_marker=None,
            os_chunk_count=os_chunk_count,
            qdrant_chunk_count=qdrant_chunk_count,
            missing_chunk_coverage=None,
        )

    text = fulltext_doc.get("text_full")
    text_str = text if isinstance(text, str) else ""
    text_len = len(text_str)
    empty_text = text_len == 0
    very_low_text = text_len > 0 and text_len < very_low_text_threshold

    extraction_mode_raw = fulltext_doc.get("extraction_mode")
    extraction_mode = (
        str(extraction_mode_raw).strip().lower()
        if extraction_mode_raw is not None
        else None
    )
    has_ocr_marker = bool(
        (extraction_mode in _OCR_EXTRACTION_MODES)
        or any(fulltext_doc.get(field) is not None for field in _OCR_MARKER_FIELDS)
    )

    missing_chunk_coverage: bool | None = None
    if os_chunk_count is not None or qdrant_chunk_count is not None:
        os_ok = os_chunk_count is not None and os_chunk_count > 0
        qdrant_ok = qdrant_chunk_count is not None and qdrant_chunk_count > 0
        missing_chunk_coverage = not (os_ok or qdrant_ok)

    return ExpectedDocIndicator(
        checksum=checksum,
        exists_in_fulltext=True,
        path=fulltext_doc.get("path"),
        filename=fulltext_doc.get("filename"),
        doc_type=fulltext_doc.get("doc_type"),
        extraction_mode=extraction_mode,
        text_length=text_len,
        empty_text=_bool_or_none(empty_text),
        very_low_text_length=_bool_or_none(very_low_text),
        has_ocr_marker=_bool_or_none(has_ocr_marker),
        os_chunk_count=os_chunk_count,
        qdrant_chunk_count=qdrant_chunk_count,
        missing_chunk_coverage=_bool_or_none(missing_chunk_coverage),
    )


def _first_expected_rank(
    checksums_in_order: Sequence[str],
    expected_checksums: Sequence[str],
) -> int | None:
    expected_set = set(expected_checksums)
    for idx, checksum in enumerate(checksums_in_order, start=1):
        if checksum in expected_set:
            return idx
    return None


def _resolve_primary_expected_checksum(
    *,
    expected_checksums: Sequence[str],
    top_checksums: Sequence[str],
    deep_probe_checksums: Sequence[str],
) -> str | None:
    if not expected_checksums:
        return None
    expected_set = set(expected_checksums)
    for checksum in top_checksums:
        if checksum in expected_set:
            return checksum
    for checksum in deep_probe_checksums:
        if checksum in expected_set:
            return checksum
    return expected_checksums[0]


def _top_doc_records(
    row: Mapping[str, Any],
    metadata_cache: dict[str, dict[str, Any] | None],
    os_client: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank in (1, 2, 3):
        checksum = row.get(f"top{rank}_checksum")
        score = row.get(f"top{rank}_score")
        metadata = None
        if isinstance(checksum, str) and checksum:
            if checksum not in metadata_cache:
                metadata_cache[checksum] = _fetch_fulltext_by_checksum(os_client, checksum)
            metadata = metadata_cache[checksum]
        records.append(
            {
                "rank": rank,
                "checksum": checksum,
                "score": score,
                "path": (metadata or {}).get("path"),
                "filename": (metadata or {}).get("filename"),
                "doc_type": (metadata or {}).get("doc_type"),
            }
        )
    return records


def _run_deep_probe(
    query: str,
    cfg: RetrievalConfig,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        output = retrieve(query, cfg=cfg)
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"
    return list(output.documents), None


def _safe_checksums(hits: Iterable[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for hit in hits:
        checksum = hit.get("checksum")
        if isinstance(checksum, str) and checksum:
            values.append(checksum)
    return values


def analyze_residual_failures(
    *,
    patha_runbook_path: Path,
    fixture_path: Path,
    output_path: Path,
    candidate_depth: int,
    very_low_text_threshold: int,
    support_labels_path: Path | None = DEFAULT_SUPPORT_LABELS_PATH,
) -> dict[str, Any]:
    patha_data = _load_json(patha_runbook_path)
    fixture_data = _load_json(fixture_path)

    fixture_queries: dict[str, dict[str, Any]] = {
        str(q.get("id")): q
        for q in fixture_data.get("queries", [])
        if isinstance(q, dict) and q.get("id")
    }

    positive_rows: list[dict[str, Any]] = [
        row
        for row in patha_data.get("rows", [])
        if isinstance(row, dict)
        and row.get("mode") == "positive"
    ]
    support_label_meta, support_label_overrides = _load_support_labels(support_labels_path)
    default_positive_query_type = _normalize_query_type(
        support_label_meta.get("default_positive_query_type")
    ) or DEFAULT_POSITIVE_QUERY_TYPE

    os_client = get_client()
    metadata_cache: dict[str, dict[str, Any] | None] = {}

    probe_cfg = _build_probe_config(patha_data.get("config", {}), candidate_depth=candidate_depth)
    per_query_rows: list[dict[str, Any]] = []
    bucket_counter: Counter[str] = Counter()
    query_class_counter: Counter[str] = Counter()
    query_type_counter: Counter[str] = Counter()
    benchmark_mode_counter: Counter[str] = Counter()
    selected_failure_mode_counter: Counter[str] = Counter()

    for row in positive_rows:
        query_id = str(row.get("query_id") or "")
        query_text = str(row.get("query") or "")
        fixture_row = fixture_queries.get(query_id, {})
        strict_expected_checksums = _dedupe_str_checksums(
            row.get("expected_checksums") or fixture_row.get("expected_checksums") or []
        )
        label_override = support_label_overrides.get(query_id)
        support_expected_checksums, _ = resolve_support_expectations(
            strict_expected_checksums=strict_expected_checksums,
            label_override=label_override,
        )
        query_type = resolve_query_benchmark_type(
            fixture_query=fixture_row,
            label_override=label_override,
            default_query_type=default_positive_query_type,
        )
        benchmark_mode = benchmark_mode_for_query_type(query_type)
        expected_doc_types = list(fixture_row.get("expected_doc_types") or [])
        expected_checksums_for_failure_lens = (
            support_expected_checksums
            if benchmark_mode == BENCHMARK_MODE_ANSWER_SUPPORT
            else strict_expected_checksums
        )

        query_class = classify_query_anchor(query_text)
        query_class_counter[query_class] += 1
        query_type_counter[query_type] += 1
        benchmark_mode_counter[benchmark_mode] += 1

        top_docs = _top_doc_records(row, metadata_cache, os_client)
        top_checksums = [
            str(doc["checksum"])
            for doc in top_docs
            if isinstance(doc.get("checksum"), str)
        ]
        strict_rank_in_top3 = _first_expected_rank(top_checksums, strict_expected_checksums)
        support_rank_in_top3 = _first_expected_rank(top_checksums, support_expected_checksums)
        strict_hit_at_1_archived = bool(row.get("hit_at_1"))
        strict_hit_at_3_top3 = (
            isinstance(strict_rank_in_top3, int) and strict_rank_in_top3 <= 3
        )
        support_hit_at_1_top3 = support_rank_in_top3 == 1
        support_hit_at_3_top3 = (
            isinstance(support_rank_in_top3, int) and support_rank_in_top3 <= 3
        )
        selected_for_failure = (
            not strict_hit_at_1_archived
            if benchmark_mode == BENCHMARK_MODE_STRICT_RETRIEVAL
            else not support_hit_at_1_top3
        )
        if not selected_for_failure:
            continue
        selected_failure_mode_counter[benchmark_mode] += 1

        deep_hits, deep_probe_error = _run_deep_probe(query_text, probe_cfg)
        deep_checksums = _safe_checksums(deep_hits)
        strict_rank_in_candidates = _first_expected_rank(deep_checksums, strict_expected_checksums)
        support_rank_in_candidates = _first_expected_rank(
            deep_checksums, support_expected_checksums
        )
        expected_rank_in_candidates = (
            support_rank_in_candidates
            if benchmark_mode == BENCHMARK_MODE_ANSWER_SUPPORT
            else strict_rank_in_candidates
        )
        relevant_in_candidates = (
            None if deep_probe_error else bool(expected_rank_in_candidates is not None)
        )

        expected_indicators: list[ExpectedDocIndicator] = []
        for checksum in expected_checksums_for_failure_lens:
            if checksum not in metadata_cache:
                metadata_cache[checksum] = _fetch_fulltext_by_checksum(os_client, checksum)
            fulltext_doc = metadata_cache[checksum]
            os_count = _count_os_chunks_by_checksum(os_client, checksum)
            qdrant_count = count_qdrant_chunks_by_checksum(checksum)
            expected_indicators.append(
                _indicator_from_fulltext(
                    checksum=checksum,
                    fulltext_doc=fulltext_doc,
                    os_chunk_count=os_count,
                    qdrant_chunk_count=qdrant_count,
                    very_low_text_threshold=very_low_text_threshold,
                )
            )

        any_expected_exists = any(ind.exists_in_fulltext for ind in expected_indicators)
        text_gap_evidence = any(
            bool(ind.empty_text)
            or bool(ind.very_low_text_length)
            or bool(ind.missing_chunk_coverage)
            for ind in expected_indicators
            if ind.exists_in_fulltext
        )
        text_available_evidence = any(
            ind.exists_in_fulltext
            and not bool(ind.empty_text)
            and not bool(ind.very_low_text_length)
            and not bool(ind.missing_chunk_coverage)
            for ind in expected_indicators
        )
        corpus_absence_evidence = not any_expected_exists

        bucket = assign_primary_bucket(
            relevant_in_top3=(
                support_hit_at_3_top3
                if benchmark_mode == BENCHMARK_MODE_ANSWER_SUPPORT
                else strict_hit_at_3_top3
            ),
            relevant_in_candidates=relevant_in_candidates,
            expected_rank_in_candidates=expected_rank_in_candidates,
            text_gap_evidence=text_gap_evidence,
            corpus_absence_evidence=corpus_absence_evidence,
            text_available_evidence=text_available_evidence,
        )
        bucket_counter[bucket] += 1

        primary_expected_checksum = _resolve_primary_expected_checksum(
            expected_checksums=expected_checksums_for_failure_lens,
            top_checksums=top_checksums,
            deep_probe_checksums=deep_checksums,
        )
        primary_indicator = None
        if primary_expected_checksum:
            for indicator in expected_indicators:
                if indicator.checksum == primary_expected_checksum:
                    primary_indicator = indicator
                    break

        per_query_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "benchmark_query_type": query_type,
                "benchmark_primary_mode": benchmark_mode,
                "query_classification": query_class,
                "expected_reference": {
                    "expected_doc_types": expected_doc_types,
                    "strict_expected_checksums": strict_expected_checksums,
                    "answer_support_checksums": support_expected_checksums,
                    "expected_checksums_for_failure_lens": expected_checksums_for_failure_lens,
                    "primary_expected_checksum": primary_expected_checksum,
                    "primary_expected_path": (
                        primary_indicator.path if primary_indicator else None
                    ),
                    "fixture_note": fixture_row.get("notes"),
                },
                "selected_for_failure_analysis": selected_for_failure,
                "strict_hit_at_1_archived": strict_hit_at_1_archived,
                "strict_relevant_doc_in_top3": strict_hit_at_3_top3,
                "answer_support_hit_at_1_top3": support_hit_at_1_top3,
                "answer_support_relevant_doc_in_top3": support_hit_at_3_top3,
                "relevant_doc_in_top3": (
                    support_hit_at_3_top3
                    if benchmark_mode == BENCHMARK_MODE_ANSWER_SUPPORT
                    else strict_hit_at_3_top3
                ),
                "relevant_doc_in_candidates": relevant_in_candidates,
                "strict_expected_rank_in_candidates": strict_rank_in_candidates,
                "answer_support_rank_in_candidates": support_rank_in_candidates,
                "expected_rank_in_candidates": expected_rank_in_candidates,
                "top_returned_docs": top_docs,
                "expected_doc_indicators": [
                    asdict(indicator) for indicator in expected_indicators
                ],
                "bucket": bucket,
                "bucket_evidence": {
                    "text_gap_evidence": text_gap_evidence,
                    "corpus_absence_evidence": corpus_absence_evidence,
                    "text_available_evidence": text_available_evidence,
                },
                "deep_probe_error": deep_probe_error,
                "reviewer_bucket_override": None,
                "reviewer_notes": "",
            }
        )

    total_failed = len(per_query_rows)
    bucket_summary = []
    for bucket in BUCKET_ORDER:
        count = int(bucket_counter.get(bucket, 0))
        bucket_summary.append(
            {
                "bucket": bucket,
                "count": count,
                "percentage": round((count / total_failed) * 100.0, 2)
                if total_failed
                else 0.0,
            }
        )

    recommendation = build_ocr_recommendation(bucket_counter, total_failed)
    query_type_summary = {
        query_type: query_type_counter.get(query_type, 0)
        for query_type in QUERY_TYPE_ORDER
    }
    benchmark_mode_summary = {
        BENCHMARK_MODE_STRICT_RETRIEVAL: benchmark_mode_counter.get(
            BENCHMARK_MODE_STRICT_RETRIEVAL, 0
        ),
        BENCHMARK_MODE_ANSWER_SUPPORT: benchmark_mode_counter.get(
            BENCHMARK_MODE_ANSWER_SUPPORT, 0
        ),
    }
    output = {
        "schema_version": RESIDUAL_FAILURE_ANALYSIS_SCHEMA_VERSION,
        "compatibility_note": (
            "Schema v2 adds explicit benchmarking metadata and query-type/mode summaries. "
            "Consumers should read aggregate_summary.*_summary fields."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": {
            "patha_runbook": str(patha_runbook_path),
            "fixture": str(fixture_path),
            "support_labels": str(support_labels_path) if support_labels_path else None,
        },
        "analysis_scope": {
            "failed_positive_queries": total_failed,
            "criterion": (
                "mode=positive with benchmark-aware failure lens: strict_retrieval hit@1 "
                "for canonical/ambiguous queries, answer_support hit@1 for explicitly "
                "multi-source queries"
            ),
        },
        "benchmarking": {
            "benchmark_modes": [
                BENCHMARK_MODE_STRICT_RETRIEVAL,
                BENCHMARK_MODE_ANSWER_SUPPORT,
            ],
            "query_type_labels": list(QUERY_TYPE_ORDER),
            "default_positive_query_type": default_positive_query_type,
            "support_labels_overrides_count": len(support_label_overrides),
        },
        "probe_method": {
            "candidate_probe": {
                "enabled": True,
                "deterministic": True,
                "description": "Exact-query deep retrieval probe (variants disabled) to avoid LLM rewrite variance.",
                "config": {
                    "top_k": probe_cfg.top_k,
                    "top_k_each": probe_cfg.top_k_each,
                    "enable_variants": probe_cfg.enable_variants,
                    "enable_mmr": probe_cfg.enable_mmr,
                    "fusion_weight_vector": probe_cfg.fusion_weight_vector,
                    "fusion_weight_bm25": probe_cfg.fusion_weight_bm25,
                },
            },
            "text_gap_thresholds": {
                "very_low_text_length_lt": very_low_text_threshold,
            },
        },
        "residual_failures": per_query_rows,
        "aggregate_summary": {
            "total_failed_positive_queries": total_failed,
            "bucket_summary": bucket_summary,
            "query_classification_summary": dict(query_class_counter),
            "benchmark_query_type_summary": query_type_summary,
            "benchmark_primary_mode_summary": benchmark_mode_summary,
            "selected_failure_mode_summary": dict(selected_failure_mode_counter),
            "deep_probe_errors": sum(
                1 for row in per_query_rows if row.get("deep_probe_error")
            ),
        },
        "ocr_canary_recommendation": recommendation,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return output


def _default_output_path(patha_runbook_path: Path) -> Path:
    return patha_runbook_path.with_name(
        f"{patha_runbook_path.stem}_residual_failure_analysis.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic residual-failure analysis for Path A retrieval misses."
    )
    parser.add_argument(
        "--patha-runbook",
        type=Path,
        default=Path("docs/runbooks/retrieval_eval_postfix_2026-03-26_patha_v1.json"),
        help="Path to Path A retrieval eval JSON artifact.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/retrieval_eval_queries.json"),
        help="Path to retrieval fixture JSON.",
    )
    parser.add_argument(
        "--support-labels",
        type=Path,
        default=DEFAULT_SUPPORT_LABELS_PATH,
        help=(
            "Optional JSON file containing manual answer-support checksums and "
            "benchmark query-type overrides."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output sidecar artifact path under docs/runbooks/.",
    )
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=DEFAULT_CANDIDATE_DEPTH,
        help="Deep-probe candidate depth for expected-checksum rank detection.",
    )
    parser.add_argument(
        "--very-low-text-threshold",
        type=int,
        default=DEFAULT_VERY_LOW_TEXT_THRESHOLD,
        help="Text length threshold below which expected-doc text is marked very low.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or _default_output_path(args.patha_runbook)
    output = analyze_residual_failures(
        patha_runbook_path=args.patha_runbook,
        fixture_path=args.fixture,
        output_path=output_path,
        candidate_depth=max(args.candidate_depth, 1),
        very_low_text_threshold=max(args.very_low_text_threshold, 1),
        support_labels_path=args.support_labels,
    )
    recommendation = output.get("ocr_canary_recommendation", {})
    print(f"Wrote: {output_path}")
    print(
        "OCR canary recommendation: "
        f"{recommendation.get('decision')} "
        f"(text-gap={recommendation.get('evidence', {}).get('text_gap_failures')}/"
        f"{recommendation.get('evidence', {}).get('total_failed_positives')})"
    )


if __name__ == "__main__":
    main()
