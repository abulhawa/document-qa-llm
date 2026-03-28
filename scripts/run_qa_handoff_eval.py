from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

# Ensure repo root is on sys.path when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.query_rewriter import has_strong_query_anchors
from core.retrieval.types import RetrievalConfig
from qa_pipeline.handoff import (
    DEFAULT_DYNAMIC_MIN_CHUNKS,
    estimate_tokens,
    pack_docs_by_token_budget,
)
from qa_pipeline.llm_client import generate_answer
from qa_pipeline.prompt_builder import build_prompt
from qa_pipeline.retrieve import retrieve_context
from qa_pipeline.rewrite import rewrite_question
from qa_pipeline.types import RetrievedDocument, RetrievalResult

DEFAULT_FIXTURE_PATH = Path("tests/fixtures/retrieval_eval_queries.json")
DEFAULT_SUPPORT_LABELS_PATH = Path("tests/fixtures/retrieval_eval_answer_support_labels.json")
DEFAULT_OUTPUT_DIR = Path("docs/runbooks")
DEFAULT_SOFT_TIMEOUT_SECONDS = 45.0
DEFAULT_DYNAMIC_TOKEN_BUDGET = 1200
DEFAULT_DYNAMIC_RETRIEVAL_TOP_K = 7

PROFILE_DOC_TYPES = {"cv", "resume", "cover_letter", "reference_letter", "profile"}
TARGET_WHEN_WHERE_TERMS = {"when", "where"}
TOKEN_RE = re.compile(r"[a-z0-9]+")
FALLBACK_ANSWERS = {
    "",
    "i don't know.",
    "no relevant context found to answer the question.",
    "❌ retrieval failed.",
    "❌ llm call failed.",
}


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    retrieval_top_k: int
    dynamic_token_budget: Optional[int] = None
    dynamic_min_chunks: int = DEFAULT_DYNAMIC_MIN_CHUNKS
    dynamic_max_chunks: Optional[int] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_date_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def _dedupe_checksums(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _load_support_overrides(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    overrides = payload.get("overrides", {})
    if not isinstance(overrides, Mapping):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw_qid, raw_override in overrides.items():
        if not isinstance(raw_qid, str) or not isinstance(raw_override, Mapping):
            continue
        out[raw_qid] = dict(raw_override)
    return out


def resolve_support_checksums(
    strict_expected_checksums: Sequence[str],
    label_override: Optional[Mapping[str, Any]],
) -> list[str]:
    strict = _dedupe_checksums(strict_expected_checksums)
    if not label_override:
        return strict

    mode = str(label_override.get("answer_support_mode") or "merge").strip().lower()
    extras = _dedupe_checksums(label_override.get("answer_support_checksums") or [])
    if mode == "replace":
        return extras
    return _dedupe_checksums([*strict, *extras])


def is_profile_when_where_query(row: Mapping[str, Any]) -> bool:
    if str(row.get("mode") or "").lower() != "positive":
        return False
    query = str(row.get("query") or "")
    tokens = _tokens(query)
    if not (tokens & TARGET_WHEN_WHERE_TERMS):
        return False

    expected_doc_types = {
        str(item).strip().lower()
        for item in (row.get("expected_doc_types") or [])
        if isinstance(item, str)
    }
    target_areas = {
        str(item).strip().lower()
        for item in (row.get("target_areas") or [])
        if isinstance(item, str)
    }
    notes = str(row.get("notes") or "").lower()
    has_profile_scope = bool(expected_doc_types & PROFILE_DOC_TYPES) or (
        "career_cv_docs" in target_areas
    ) or ("profile" in notes)
    return has_profile_scope


def _normalize_doc_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _intended_doc_types_for_integrity(row: Mapping[str, Any]) -> list[str]:
    expected_doc_types = sorted(
        {
            _normalize_doc_type(item)
            for item in (row.get("expected_doc_types") or [])
            if isinstance(item, str) and _normalize_doc_type(item) not in {"", "__missing__"}
        }
    )
    if expected_doc_types:
        return expected_doc_types

    target_areas = {
        str(item).strip().lower()
        for item in (row.get("target_areas") or [])
        if isinstance(item, str)
    }
    if "career_cv_docs" in target_areas:
        return ["cv", "resume"]
    return []


def _default_fulltext_fetcher(checksum: str) -> Optional[dict[str, Any]]:
    try:
        from utils.opensearch_utils import get_fulltext_by_checksum
    except Exception:
        return None

    try:
        payload = get_fulltext_by_checksum(checksum)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def build_benchmark_integrity_diagnostics(
    fixture_rows: Sequence[Mapping[str, Any]],
    *,
    fulltext_fetcher: Optional[Callable[[str], Optional[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    fetcher = fulltext_fetcher or _default_fulltext_fetcher
    metadata_cache: dict[str, Optional[dict[str, Any]]] = {}

    query_diagnostics: dict[str, dict[str, Any]] = {}
    status_counts = {
        "pass": 0,
        "drift": 0,
        "unresolved": 0,
        "not_applicable": 0,
    }
    integrity_failure_query_ids: list[str] = []
    checked_positive_queries = 0

    for row in fixture_rows:
        query_id = str(row.get("id") or "")
        mode = str(row.get("mode") or "").strip().lower()
        expected_checksums = _dedupe_checksums(row.get("expected_checksums") or [])
        intended_doc_types = _intended_doc_types_for_integrity(row)

        status = "not_applicable"
        reason = "query not in integrity check scope"
        resolved_checksums: list[str] = []
        matching_checksums: list[str] = []
        missing_checksums: list[str] = []
        checksum_diagnostics: list[dict[str, Any]] = []

        if mode != "positive":
            reason = "non-positive benchmark mode"
        elif not expected_checksums:
            reason = "no expected checksums"
        elif not intended_doc_types:
            reason = "no intended doc family in fixture metadata"
        else:
            checked_positive_queries += 1
            for checksum in expected_checksums:
                if checksum not in metadata_cache:
                    metadata_cache[checksum] = fetcher(checksum)
                metadata = metadata_cache[checksum] or {}
                doc_type = _normalize_doc_type(metadata.get("doc_type"))
                path = str(metadata.get("path") or "")
                filename = str(metadata.get("filename") or "")
                exists = bool(metadata)
                matches_family = exists and (doc_type in intended_doc_types)

                if exists:
                    resolved_checksums.append(checksum)
                else:
                    missing_checksums.append(checksum)
                if matches_family:
                    matching_checksums.append(checksum)

                checksum_diagnostics.append(
                    {
                        "checksum": checksum,
                        "exists_in_snapshot": exists,
                        "doc_type": doc_type or None,
                        "path": path or None,
                        "filename": filename or None,
                        "matches_intended_family": matches_family,
                    }
                )

            if matching_checksums:
                status = "pass"
                reason = "at least one expected checksum resolves to intended doc family"
            elif resolved_checksums:
                status = "drift"
                reason = "expected checksums resolve, but not to intended doc family"
            else:
                status = "unresolved"
                reason = "expected checksums not found in current index snapshot"

        if status in {"drift", "unresolved"} and query_id:
            integrity_failure_query_ids.append(query_id)
        status_counts[status] += 1

        query_diagnostics[query_id] = {
            "query_id": query_id,
            "mode": mode,
            "expected_checksums": expected_checksums,
            "intended_doc_types": intended_doc_types,
            "status": status,
            "reason": reason,
            "integrity_failure": status in {"drift", "unresolved"},
            "resolved_checksums": resolved_checksums,
            "matching_checksums": matching_checksums,
            "missing_checksums": missing_checksums,
            "checksums": checksum_diagnostics,
        }

    return {
        "generated_at_utc": _utc_now_iso(),
        "summary": {
            "checked_positive_queries": checked_positive_queries,
            "pass_count": status_counts["pass"],
            "drift_count": status_counts["drift"],
            "unresolved_count": status_counts["unresolved"],
            "not_applicable_count": status_counts["not_applicable"],
            "integrity_failure_count": len(integrity_failure_query_ids),
            "integrity_failure_query_ids": integrity_failure_query_ids,
        },
        "queries": query_diagnostics,
    }


def _expected_rank(docs: Sequence[RetrievedDocument], expected_checksums: Sequence[str]) -> Optional[int]:
    expected = {item for item in expected_checksums if item}
    if not expected:
        return None
    for idx, doc in enumerate(docs, start=1):
        checksum = str(doc.checksum or "")
        if checksum and checksum in expected:
            return idx
    return None


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(1.0, pct / 100.0)) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _normalize_answer(answer: str) -> str:
    return (answer or "").strip().lower()


def _is_error_answer(answer: str) -> bool:
    normalized = (answer or "").strip()
    return normalized.startswith("[LLM Error:") or normalized.startswith("❌")


def _is_fallback_answer(answer: str) -> bool:
    normalized = _normalize_answer(answer)
    if normalized in FALLBACK_ANSWERS:
        return True
    return normalized.startswith("**clarify**")


def _base_cfg(top_k: int, top_k_each: int) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=max(top_k, 1),
        top_k_each=max(top_k_each, 1),
        enable_variants=True,
        enable_mmr=True,
        sibling_expansion_enabled=True,
        enable_rerank=False,
    )


def _build_clarify_answer(clarification: str) -> str:
    return f"**Clarify**:  \n   -  {clarification}.  \n\nTry again!"


def _run_query(
    row: Mapping[str, Any],
    *,
    strategy: StrategyConfig,
    support_overrides: Mapping[str, Mapping[str, Any]],
    benchmark_integrity_by_query_id: Mapping[str, Mapping[str, Any]],
    run_date: str,
    top_k_each: int,
    soft_timeout_seconds: float,
    mode: str,
    temperature: float,
    use_cache: bool,
) -> dict[str, Any]:
    query_id = str(row.get("id") or "")
    query_text = str(row.get("query") or "")
    query_mode = str(row.get("mode") or "")
    strict_expected = _dedupe_checksums(row.get("expected_checksums") or [])
    support_expected = resolve_support_checksums(strict_expected, support_overrides.get(query_id))
    targeted = is_profile_when_where_query(row)
    integrity_diag = benchmark_integrity_by_query_id.get(query_id, {})
    integrity_status = str(integrity_diag.get("status") or "not_applicable")
    integrity_reason = str(integrity_diag.get("reason") or "")
    integrity_failure = bool(integrity_diag.get("integrity_failure"))
    intended_doc_types = [
        str(item)
        for item in (integrity_diag.get("intended_doc_types") or [])
        if isinstance(item, str)
    ]

    query_start = time.perf_counter()
    error: Optional[str] = None
    clarification: Optional[str] = None
    answer = ""
    packed_docs: list[RetrievedDocument] = []
    retrieved_docs_count = 0
    packed_tokens = 0
    packing_duration_ms = 0.0
    rewritten_question = query_text
    prompt_mode = mode

    try:
        rewrite_result = rewrite_question(
            query_text,
            temperature=0.15,
            use_cache=use_cache,
        )
        rewritten_question = str(rewrite_result.rewritten or "")
        clarification = str(rewrite_result.clarify or "") or None

        if has_strong_query_anchors(query_text):
            rewritten_question = query_text
        if clarification and has_strong_query_anchors(query_text):
            clarification = None
            rewritten_question = query_text

        if clarification:
            answer = _build_clarify_answer(clarification)
        elif not rewritten_question:
            answer = "❌ Unexpected error occurred... ERR-QRWR"
        else:
            cfg = _base_cfg(strategy.retrieval_top_k, top_k_each)
            retrieval = retrieve_context(
                rewritten_question,
                top_k=strategy.retrieval_top_k,
                retrieval_cfg=cfg,
            )
            retrieved_docs_count = len(retrieval.documents)

            if strategy.dynamic_token_budget is not None:
                pack_start = time.perf_counter()
                packed_docs, packed_tokens = pack_docs_by_token_budget(
                    retrieval.documents,
                    token_budget=strategy.dynamic_token_budget,
                    min_chunks=strategy.dynamic_min_chunks,
                    max_chunks=strategy.dynamic_max_chunks,
                )
                packing_duration_ms = (time.perf_counter() - pack_start) * 1000.0
            else:
                packed_docs = list(retrieval.documents)
                packed_tokens = sum(estimate_tokens(doc.text) for doc in packed_docs)

            if not packed_docs:
                answer = "No relevant context found to answer the question."
            else:
                prompt_retrieval = RetrievalResult(query=retrieval.query, documents=packed_docs)
                prompt_request = build_prompt(
                    prompt_retrieval,
                    query_text,
                    mode=prompt_mode,
                    chat_history=[],
                )
                answer = generate_answer(
                    prompt_request=prompt_request,
                    temperature=temperature,
                    model=None,
                    use_cache=use_cache,
                )
    except Exception as exc:  # noqa: BLE001
        error = f"{exc.__class__.__name__}: {exc}"
        if not answer:
            answer = "❌ Unexpected error occurred... ERR-PIPELINE"

    query_duration_ms = (time.perf_counter() - query_start) * 1000.0
    timeout_exceeded = (
        soft_timeout_seconds > 0.0
        and query_duration_ms > (soft_timeout_seconds * 1000.0)
    )

    strict_rank = _expected_rank(packed_docs, strict_expected)
    support_rank = _expected_rank(packed_docs, support_expected)
    strict_hit = isinstance(strict_rank, int)
    support_hit = isinstance(support_rank, int)

    is_error = _is_error_answer(answer) or bool(error)
    is_fallback = _is_fallback_answer(answer)
    answered_without_error = (not is_error) and (not is_fallback)

    return {
        "run_date": run_date,
        "strategy": strategy.name,
        "query_id": query_id,
        "mode": query_mode,
        "query": query_text,
        "rewritten_question": rewritten_question,
        "target_profile_when_where": targeted,
        "benchmark_integrity_status": integrity_status,
        "benchmark_integrity_reason": integrity_reason,
        "benchmark_integrity_failure": integrity_failure,
        "benchmark_intended_doc_types": intended_doc_types,
        "expected_checksums": strict_expected,
        "support_expected_checksums": support_expected,
        "retrieval_top_k": strategy.retrieval_top_k,
        "dynamic_token_budget": strategy.dynamic_token_budget,
        "retrieved_docs_count": retrieved_docs_count,
        "packed_docs_count": len(packed_docs),
        "packed_context_tokens_est": packed_tokens,
        "packing_duration_ms": round(packing_duration_ms, 3),
        "packed_checksums": [str(doc.checksum or "") for doc in packed_docs],
        "strict_rank": strict_rank,
        "support_rank": support_rank,
        "strict_context_hit": strict_hit,
        "support_context_hit": support_hit,
        "answer": answer,
        "is_error": is_error,
        "is_fallback": is_fallback,
        "answered_without_error": answered_without_error,
        "clarification": clarification,
        "query_duration_ms": round(query_duration_ms, 3),
        "timeout_exceeded": timeout_exceeded,
        "error": error,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_rows = len(rows)
    positive_rows = [row for row in rows if row.get("mode") == "positive"]
    control_rows = [row for row in rows if row.get("mode") == "control"]
    targeted_rows = [row for row in positive_rows if bool(row.get("target_profile_when_where"))]
    errored_rows = [row for row in rows if bool(row.get("is_error")) or bool(row.get("error"))]
    fallback_rows = [row for row in rows if bool(row.get("is_fallback"))]
    timeout_rows = [row for row in rows if bool(row.get("timeout_exceeded"))]

    positive_total = len(positive_rows)
    positive_answered = sum(1 for row in positive_rows if bool(row.get("answered_without_error")))
    positive_support_hits = sum(1 for row in positive_rows if bool(row.get("support_context_hit")))
    positive_strict_hits = sum(1 for row in positive_rows if bool(row.get("strict_context_hit")))
    positive_answered_with_support = sum(
        1
        for row in positive_rows
        if bool(row.get("answered_without_error")) and bool(row.get("support_context_hit"))
    )

    integrity_failure_rows = [
        row for row in positive_rows if bool(row.get("benchmark_integrity_failure"))
    ]
    integrity_clean_rows = [
        row for row in positive_rows if not bool(row.get("benchmark_integrity_failure"))
    ]
    integrity_clean_total = len(integrity_clean_rows)
    integrity_clean_answered = sum(
        1 for row in integrity_clean_rows if bool(row.get("answered_without_error"))
    )
    integrity_clean_support_hits = sum(
        1 for row in integrity_clean_rows if bool(row.get("support_context_hit"))
    )
    integrity_clean_answered_with_support = sum(
        1
        for row in integrity_clean_rows
        if bool(row.get("answered_without_error")) and bool(row.get("support_context_hit"))
    )
    retrieval_failure_rows = [
        row for row in integrity_clean_rows if not bool(row.get("support_context_hit"))
    ]

    targeted_total = len(targeted_rows)
    targeted_answered = sum(1 for row in targeted_rows if bool(row.get("answered_without_error")))
    targeted_support_hits = sum(1 for row in targeted_rows if bool(row.get("support_context_hit")))

    control_total = len(control_rows)
    control_answered = sum(1 for row in control_rows if bool(row.get("answered_without_error")))
    control_fallback = sum(1 for row in control_rows if bool(row.get("is_fallback")))

    duration_samples = [float(row.get("query_duration_ms") or 0.0) for row in rows]
    avg_duration = (sum(duration_samples) / total_rows) if total_rows else 0.0
    p95_duration = _percentile(duration_samples, 95.0) if duration_samples else 0.0
    avg_packing_duration = (
        sum(float(row.get("packing_duration_ms") or 0.0) for row in rows) / total_rows
        if total_rows
        else 0.0
    )
    avg_packed_docs = (
        sum(int(row.get("packed_docs_count") or 0) for row in rows) / total_rows
        if total_rows
        else 0.0
    )
    avg_packed_tokens = (
        sum(float(row.get("packed_context_tokens_est") or 0.0) for row in rows) / total_rows
        if total_rows
        else 0.0
    )

    positive_error_count = sum(
        1 for row in positive_rows if bool(row.get("is_error")) or bool(row.get("error"))
    )
    positive_fallback_count = sum(1 for row in positive_rows if bool(row.get("is_fallback")))
    overall_error_rate = (len(errored_rows) / total_rows) if total_rows else 0.0
    overall_fallback_rate = (len(fallback_rows) / total_rows) if total_rows else 0.0
    positive_error_rate = (positive_error_count / positive_total) if positive_total else 0.0
    positive_fallback_rate = (positive_fallback_count / positive_total) if positive_total else 0.0

    return {
        "positive_total": positive_total,
        "positive_answered_without_error": positive_answered,
        "positive_answered_without_error_rate": round((positive_answered / positive_total), 4)
        if positive_total
        else 0.0,
        "positive_support_context_hits": positive_support_hits,
        "positive_support_context_hit_rate": round((positive_support_hits / positive_total), 4)
        if positive_total
        else 0.0,
        "positive_strict_context_hits": positive_strict_hits,
        "positive_strict_context_hit_rate": round((positive_strict_hits / positive_total), 4)
        if positive_total
        else 0.0,
        "positive_answered_with_support": positive_answered_with_support,
        "positive_answered_with_support_rate": round(
            (positive_answered_with_support / positive_total), 4
        )
        if positive_total
        else 0.0,
        "positive_integrity_failures": len(integrity_failure_rows),
        "positive_integrity_failure_query_ids": [
            str(row.get("query_id") or "") for row in integrity_failure_rows
        ],
        "positive_integrity_clean_total": integrity_clean_total,
        "positive_integrity_clean_answered_without_error": integrity_clean_answered,
        "positive_integrity_clean_answered_without_error_rate": round(
            (integrity_clean_answered / integrity_clean_total), 4
        )
        if integrity_clean_total
        else 0.0,
        "positive_integrity_clean_support_context_hits": integrity_clean_support_hits,
        "positive_integrity_clean_support_context_hit_rate": round(
            (integrity_clean_support_hits / integrity_clean_total), 4
        )
        if integrity_clean_total
        else 0.0,
        "positive_integrity_clean_answered_with_support": integrity_clean_answered_with_support,
        "positive_integrity_clean_answered_with_support_rate": round(
            (integrity_clean_answered_with_support / integrity_clean_total), 4
        )
        if integrity_clean_total
        else 0.0,
        "positive_true_retrieval_failures_excluding_integrity": len(retrieval_failure_rows),
        "positive_true_retrieval_failure_query_ids": [
            str(row.get("query_id") or "") for row in retrieval_failure_rows
        ],
        "target_profile_when_where_total": targeted_total,
        "target_profile_when_where_query_ids": [
            str(row.get("query_id") or "") for row in targeted_rows
        ],
        "target_profile_when_where_answered_without_error": targeted_answered,
        "target_profile_when_where_answered_without_error_rate": round(
            (targeted_answered / targeted_total), 4
        )
        if targeted_total
        else 0.0,
        "target_profile_when_where_support_hits": targeted_support_hits,
        "target_profile_when_where_support_hit_rate": round(
            (targeted_support_hits / targeted_total), 4
        )
        if targeted_total
        else 0.0,
        "control_total": control_total,
        "control_answered_without_error": control_answered,
        "control_fallback_count": control_fallback,
        "queries_with_errors": len(errored_rows),
        "query_error_ids": [str(row.get("query_id") or "") for row in errored_rows],
        "queries_exceeding_soft_timeout": len(timeout_rows),
        "timeout_query_ids": [str(row.get("query_id") or "") for row in timeout_rows],
        "overall_error_rate": round(overall_error_rate, 4),
        "overall_fallback_rate": round(overall_fallback_rate, 4),
        "positive_error_rate": round(positive_error_rate, 4),
        "positive_fallback_rate": round(positive_fallback_rate, 4),
        "avg_query_duration_ms": round(avg_duration, 3),
        "p95_query_duration_ms": round(p95_duration, 3),
        "avg_packing_duration_ms": round(avg_packing_duration, 3),
        "avg_packed_docs_count": round(avg_packed_docs, 3),
        "avg_packed_context_tokens_est": round(avg_packed_tokens, 3),
        "avg_chunks_passed": round(avg_packed_docs, 3),
        "avg_token_usage_est": round(avg_packed_tokens, 3),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "run_date",
        "strategy",
        "query_id",
        "mode",
        "query",
        "rewritten_question",
        "target_profile_when_where",
        "benchmark_integrity_status",
        "benchmark_integrity_failure",
        "benchmark_integrity_reason",
        "retrieval_top_k",
        "dynamic_token_budget",
        "retrieved_docs_count",
        "packed_docs_count",
        "packed_context_tokens_est",
        "packing_duration_ms",
        "strict_rank",
        "support_rank",
        "strict_context_hit",
        "support_context_hit",
        "answered_without_error",
        "is_error",
        "is_fallback",
        "query_duration_ms",
        "timeout_exceeded",
        "error",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _strategy_from_name(
    name: str,
    *,
    dynamic_retrieval_top_k: int,
    dynamic_token_budget: int,
    dynamic_min_chunks: int,
) -> StrategyConfig:
    normalized = name.strip().lower()
    if normalized == "top3":
        return StrategyConfig(name="top3", retrieval_top_k=3)
    if normalized == "top5":
        return StrategyConfig(name="top5", retrieval_top_k=5)
    if normalized == "dynamic":
        return StrategyConfig(
            name="dynamic",
            retrieval_top_k=max(dynamic_retrieval_top_k, 1),
            dynamic_token_budget=max(dynamic_token_budget, 1),
            dynamic_min_chunks=max(dynamic_min_chunks, 0),
        )
    raise ValueError(f"Unsupported strategy '{name}'. Use top3, top5, dynamic.")


def _derive_output_base(user_output: Optional[Path]) -> Path:
    if user_output is not None:
        return user_output
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return DEFAULT_OUTPUT_DIR / f"qa_handoff_eval_compare_{stamp}.json"


def _run_strategy(
    fixture_rows: Sequence[Mapping[str, Any]],
    *,
    strategy: StrategyConfig,
    support_overrides: Mapping[str, Mapping[str, Any]],
    benchmark_integrity_by_query_id: Mapping[str, Mapping[str, Any]],
    benchmark_integrity_summary: Mapping[str, Any],
    top_k_each: int,
    soft_timeout_seconds: float,
    mode: str,
    temperature: float,
    use_cache: bool,
    per_query_sleep_seconds: float,
) -> dict[str, Any]:
    run_date = _iso_date_utc()
    run_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for row in fixture_rows:
        rows.append(
            _run_query(
                row,
                strategy=strategy,
                support_overrides=support_overrides,
                benchmark_integrity_by_query_id=benchmark_integrity_by_query_id,
                run_date=run_date,
                top_k_each=top_k_each,
                soft_timeout_seconds=soft_timeout_seconds,
                mode=mode,
                temperature=temperature,
                use_cache=use_cache,
            )
        )
        if per_query_sleep_seconds > 0.0:
            time.sleep(per_query_sleep_seconds)
    run_duration_ms = (time.perf_counter() - run_start) * 1000.0
    return {
        "generated_at_utc": _utc_now_iso(),
        "strategy": strategy.name,
        "config": {
            "retrieval_top_k": strategy.retrieval_top_k,
            "top_k_each": top_k_each,
            "sibling_expansion_enabled": True,
            "enable_rerank": False,
            "dynamic_token_budget": strategy.dynamic_token_budget,
            "dynamic_min_chunks": strategy.dynamic_min_chunks,
        },
        "summary": _summarize(rows),
        "benchmark_integrity_summary": dict(benchmark_integrity_summary),
        "rows": rows,
        "run_duration_ms": round(run_duration_ms, 3),
    }


def _delta(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _delta_float(after: Mapping[str, Any], before: Mapping[str, Any], key: str) -> float:
    return round(float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0), 4)


def _compare_strategies(strategy_artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    top3 = strategy_artifacts.get("top3", {})
    top5 = strategy_artifacts.get("top5", {})
    dynamic = strategy_artifacts.get("dynamic", {})

    top3_summary = top3.get("summary", {}) if isinstance(top3, Mapping) else {}
    top5_summary = top5.get("summary", {}) if isinstance(top5, Mapping) else {}
    dynamic_summary = dynamic.get("summary", {}) if isinstance(dynamic, Mapping) else {}

    return {
        "top5_vs_top3": {
            "positive_answered_without_error_delta": _delta(
                top5_summary,
                top3_summary,
                "positive_answered_without_error",
            ),
            "positive_support_context_hits_delta": _delta(
                top5_summary,
                top3_summary,
                "positive_support_context_hits",
            ),
            "positive_answered_with_support_delta": _delta(
                top5_summary,
                top3_summary,
                "positive_answered_with_support",
            ),
        },
        "dynamic_vs_top5": {
            "positive_answered_without_error_delta": _delta(
                dynamic_summary,
                top5_summary,
                "positive_answered_without_error",
            ),
            "positive_support_context_hits_delta": _delta(
                dynamic_summary,
                top5_summary,
                "positive_support_context_hits",
            ),
            "positive_answered_with_support_delta": _delta(
                dynamic_summary,
                top5_summary,
                "positive_answered_with_support",
            ),
        },
        "operational_dynamic_vs_top5": {
            "avg_chunks_passed_delta": _delta_float(
                dynamic_summary,
                top5_summary,
                "avg_chunks_passed",
            ),
            "avg_token_usage_est_delta": _delta_float(
                dynamic_summary,
                top5_summary,
                "avg_token_usage_est",
            ),
            "avg_query_duration_ms_delta": _delta_float(
                dynamic_summary,
                top5_summary,
                "avg_query_duration_ms",
            ),
            "p95_query_duration_ms_delta": _delta_float(
                dynamic_summary,
                top5_summary,
                "p95_query_duration_ms",
            ),
            "overall_error_rate_delta": _delta_float(
                dynamic_summary,
                top5_summary,
                "overall_error_rate",
            ),
            "overall_fallback_rate_delta": _delta_float(
                dynamic_summary,
                top5_summary,
                "overall_fallback_rate",
            ),
            "positive_integrity_failures_delta": _delta(
                dynamic_summary,
                top5_summary,
                "positive_integrity_failures",
            ),
        },
        "run_duration_ms": {
            name: artifact.get("run_duration_ms", 0)
            for name, artifact in strategy_artifacts.items()
            if isinstance(artifact, Mapping)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare end-to-end QA handoff policies (top3 vs top5 vs token-budgeted dynamic packing) "
            "with sibling expansion ON, rerank OFF, and benchmark-integrity diagnostics."
        )
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--support-labels", type=Path, default=DEFAULT_SUPPORT_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strategies", type=str, default="top3,top5,dynamic")
    parser.add_argument("--top-k-each", type=int, default=20)
    parser.add_argument("--dynamic-token-budget", type=int, default=DEFAULT_DYNAMIC_TOKEN_BUDGET)
    parser.add_argument("--dynamic-min-chunks", type=int, default=DEFAULT_DYNAMIC_MIN_CHUNKS)
    parser.add_argument("--dynamic-retrieval-top-k", type=int, default=DEFAULT_DYNAMIC_RETRIEVAL_TOP_K)
    parser.add_argument("--mode", type=str, default="completion")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--per-query-sleep-seconds",
        type=float,
        default=0.0,
        help="Optional throttle sleep after each query to reduce external LLM rate-limit errors.",
    )
    parser.add_argument(
        "--soft-query-timeout-seconds",
        type=float,
        default=DEFAULT_SOFT_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_base = _derive_output_base(args.output)

    fixture_payload = _load_json(args.fixture)
    fixture_rows_raw = fixture_payload.get("queries", [])
    fixture_rows: list[dict[str, Any]] = [
        dict(row) for row in fixture_rows_raw if isinstance(row, Mapping)
    ]
    support_overrides = _load_support_overrides(args.support_labels)
    benchmark_integrity = build_benchmark_integrity_diagnostics(fixture_rows)
    benchmark_integrity_by_query_id = benchmark_integrity.get("queries", {})
    benchmark_integrity_summary = benchmark_integrity.get("summary", {})

    strategy_names = [part.strip() for part in str(args.strategies).split(",") if part.strip()]
    strategies = [
        _strategy_from_name(
            name,
            dynamic_retrieval_top_k=args.dynamic_retrieval_top_k,
            dynamic_token_budget=args.dynamic_token_budget,
            dynamic_min_chunks=args.dynamic_min_chunks,
        )
        for name in strategy_names
    ]

    strategy_artifacts: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        artifact = _run_strategy(
            fixture_rows,
            strategy=strategy,
            support_overrides=support_overrides,
            benchmark_integrity_by_query_id=benchmark_integrity_by_query_id,
            benchmark_integrity_summary=benchmark_integrity_summary,
            top_k_each=max(int(args.top_k_each), 1),
            soft_timeout_seconds=max(float(args.soft_query_timeout_seconds), 0.0),
            mode=str(args.mode),
            temperature=float(args.temperature),
            use_cache=not bool(args.no_cache),
            per_query_sleep_seconds=max(float(args.per_query_sleep_seconds), 0.0),
        )
        strategy_artifacts[strategy.name] = artifact

        runbook_json_path = output_base.with_name(f"{output_base.stem}_{strategy.name}.json")
        runbook_csv_path = output_base.with_name(f"{output_base.stem}_{strategy.name}.csv")
        _write_json(runbook_json_path, artifact)
        _write_csv(runbook_csv_path, artifact.get("rows", []))
        print(f"Wrote runbook JSON ({strategy.name}): {runbook_json_path}")
        print(f"Wrote runbook CSV ({strategy.name}): {runbook_csv_path}")

    comparison_payload = {
        "generated_at_utc": _utc_now_iso(),
        "fixture": str(args.fixture),
        "support_labels": str(args.support_labels),
        "strategies": [strategy.name for strategy in strategies],
        "benchmark_integrity": benchmark_integrity,
        "comparison": _compare_strategies(strategy_artifacts),
        "artifacts": {
            name: {
                "json": str(output_base.with_name(f"{output_base.stem}_{name}.json")),
                "csv": str(output_base.with_name(f"{output_base.stem}_{name}.csv")),
            }
            for name in strategy_artifacts
        },
    }
    _write_json(output_base, comparison_payload)
    print(f"Wrote comparison JSON: {output_base}")


if __name__ == "__main__":
    main()
