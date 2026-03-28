from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

# Ensure repo root is on sys.path when run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.embeddings import embed_texts
from core.opensearch_store import (
    fetch_sibling_chunks as default_sibling_fetcher,
    search as default_keyword_retriever,
)
from core.query_rewriter import build_query_plan
from core.retrieval.pipeline import retrieve
from core.retrieval.reranker import build_configured_reranker
from core.retrieval.types import DocHit, RetrievalConfig, RetrievalDeps
from core.vector_store import retrieve_top_k as default_semantic_retriever

PROFILE_DOC_TYPES = {"cv", "resume", "cover_letter", "reference_letter", "profile"}
TARGET_WHEN_WHERE_TERMS = {"when", "where"}
TOKEN_RE = re.compile(r"[a-z0-9]+")
DEFAULT_FIXTURE_PATH = Path("tests/fixtures/retrieval_eval_queries.json")
DEFAULT_SUPPORT_LABELS_PATH = Path("tests/fixtures/retrieval_eval_answer_support_labels.json")
DEFAULT_OUTPUT_DIR = Path("docs/runbooks")
DEFAULT_SOFT_TIMEOUT_SECONDS = 30.0

CSV_BASE_COLUMNS = [
    "run_date",
    "query_id",
    "mode",
    "query",
    "expected_checksums",
    "top1_checksum",
    "top2_checksum",
    "top3_checksum",
    "top1_score",
    "top2_score",
    "top3_score",
    "hit_at_1",
    "hit_at_3",
    "clarify",
    "error",
]


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


@dataclass
class StageTimer:
    totals_ms: MutableMapping[str, float] = field(default_factory=lambda: defaultdict(float))
    call_counts: MutableMapping[str, int] = field(default_factory=lambda: defaultdict(int))
    error_counts: MutableMapping[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, stage: str, elapsed_ms: float, errored: bool = False) -> None:
        self.totals_ms[stage] += elapsed_ms
        self.call_counts[stage] += 1
        if errored:
            self.error_counts[stage] += 1


def _timed_call(
    stage_timer: StageTimer,
    stage: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    start = time.perf_counter()
    errored = False
    try:
        return fn(*args, **kwargs)
    except Exception:
        errored = True
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stage_timer.record(stage, elapsed_ms, errored=errored)


class TimedReranker:
    def __init__(self, base_reranker: Any, stage_timer: StageTimer) -> None:
        self._base = base_reranker
        self._stage_timer = stage_timer

    def rerank(
        self, query: str, docs: Sequence[DocHit], top_n: Optional[int] = None
    ) -> list[DocHit]:
        return _timed_call(
            self._stage_timer,
            "rerank",
            self._base.rerank,
            query,
            docs,
            top_n=top_n,
        )


def _build_timed_deps(stage_timer: StageTimer, base_reranker: Any) -> RetrievalDeps:
    timed_reranker = TimedReranker(base_reranker, stage_timer) if base_reranker is not None else None
    return RetrievalDeps(
        semantic_retriever=lambda query, top_k: _timed_call(
            stage_timer,
            "semantic_retriever",
            default_semantic_retriever,
            query,
            top_k,
        ),
        keyword_retriever=lambda query, top_k: _timed_call(
            stage_timer,
            "keyword_retriever",
            default_keyword_retriever,
            query,
            top_k,
        ),
        embed_texts=embed_texts,
        cross_encoder=timed_reranker,
        sibling_chunk_fetcher=lambda doc, limit: _timed_call(
            stage_timer,
            "sibling_fetch",
            default_sibling_fetcher,
            doc,
            limit,
        ),
    )


def _expected_rank(docs: Sequence[DocHit], expected_checksums: Sequence[str]) -> Optional[int]:
    expected = {item for item in expected_checksums if item}
    if not expected:
        return None
    for idx, doc in enumerate(docs, start=1):
        checksum = doc.get("checksum")
        if isinstance(checksum, str) and checksum in expected:
            return idx
    return None


def _top_checksum(docs: Sequence[DocHit], idx: int) -> str:
    if idx >= len(docs):
        return ""
    raw = docs[idx].get("checksum")
    return str(raw) if isinstance(raw, str) else ""


def _top_score(docs: Sequence[DocHit], idx: int) -> Optional[float]:
    if idx >= len(docs):
        return None
    value = docs[idx].get("retrieval_score")
    if value is None:
        value = docs[idx].get("score")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _stage_dict_with_meta(
    stage_timer: StageTimer,
    retrieve_total_ms: float,
    harness_overhead_ms: float,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    timings = {stage: round(float(ms), 3) for stage, ms in stage_timer.totals_ms.items()}
    timings["retrieve_total"] = round(float(retrieve_total_ms), 3)
    timings["harness_overhead"] = round(float(harness_overhead_ms), 3)
    calls = {stage: int(count) for stage, count in stage_timer.call_counts.items()}
    errors = {stage: int(count) for stage, count in stage_timer.error_counts.items()}
    return timings, calls, errors


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


def aggregate_stage_timings(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage_samples: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        timings = row.get("stage_timings_ms", {})
        if not isinstance(timings, Mapping):
            continue
        for stage, raw_value in timings.items():
            if isinstance(stage, str) and isinstance(raw_value, (int, float)):
                stage_samples[stage].append(float(raw_value))

    totals_ms: dict[str, float] = {}
    avg_ms: dict[str, float] = {}
    p95_ms: dict[str, float] = {}
    for stage, samples in stage_samples.items():
        totals_ms[stage] = round(sum(samples), 3)
        avg_ms[stage] = round((sum(samples) / len(samples)), 3) if samples else 0.0
        p95_ms[stage] = round(_percentile(samples, 95.0), 3)

    stage_groups_totals_ms = {
        "live_index_queries": round(
            totals_ms.get("semantic_retriever", 0.0)
            + totals_ms.get("keyword_retriever", 0.0),
            3,
        ),
        "rerank_calls": round(totals_ms.get("rerank", 0.0), 3),
        "context_expansion": round(totals_ms.get("sibling_fetch", 0.0), 3),
        "harness_overhead": round(totals_ms.get("harness_overhead", 0.0), 3),
    }
    dominant_group = max(
        stage_groups_totals_ms,
        key=lambda name: stage_groups_totals_ms.get(name, 0.0),
    ) if stage_groups_totals_ms else ""

    return {
        "stage_totals_ms": totals_ms,
        "stage_avg_ms": avg_ms,
        "stage_p95_ms": p95_ms,
        "stage_group_totals_ms": stage_groups_totals_ms,
        "dominant_stage_group": dominant_group,
    }


def _cfg_payload(cfg: RetrievalConfig) -> dict[str, Any]:
    return {
        "top_k": cfg.top_k,
        "top_k_each": cfg.top_k_each,
        "enable_variants": cfg.enable_variants,
        "enable_query_planning": cfg.enable_query_planning,
        "enable_hyde": cfg.enable_hyde,
        "enable_mmr": cfg.enable_mmr,
        "enable_rerank": cfg.enable_rerank,
        "rerank_top_n": cfg.rerank_top_n,
        "rerank_candidate_pool": cfg.rerank_candidate_pool,
        "anchored_exact_only": cfg.anchored_exact_only,
        "anchored_lexical_bias_enabled": cfg.anchored_lexical_bias_enabled,
        "anchored_fusion_weight_vector": cfg.anchored_fusion_weight_vector,
        "anchored_fusion_weight_bm25": cfg.anchored_fusion_weight_bm25,
        "sibling_expansion_enabled": cfg.sibling_expansion_enabled,
    }


def _evaluate_single_query(
    row: Mapping[str, Any],
    *,
    cfg: RetrievalConfig,
    run_date: str,
    support_overrides: Mapping[str, Mapping[str, Any]],
    soft_timeout_seconds: float,
    base_reranker: Any,
) -> dict[str, Any]:
    query_id = str(row.get("id") or "")
    mode = str(row.get("mode") or "")
    query_text = str(row.get("query") or "")
    strict_expected = _dedupe_checksums(row.get("expected_checksums") or [])
    label_override = support_overrides.get(query_id)
    support_expected = resolve_support_checksums(strict_expected, label_override)
    is_targeted_query = is_profile_when_where_query(row)

    stage_timer = StageTimer()
    deps = _build_timed_deps(stage_timer, base_reranker)

    query_start = time.perf_counter()
    retrieve_start = time.perf_counter()
    retrieval_error: Optional[str] = None
    docs: list[DocHit] = []
    clarify: Optional[str] = None
    try:
        query_plan = None
        retrieve_query = query_text
        if cfg.enable_query_planning:
            query_plan = build_query_plan(
                query_text,
                temperature=0.15,
                use_cache=True,
                enable_hyde=cfg.enable_hyde,
            )
            retrieve_query = query_plan.raw_query
        retrieval = retrieve(
            retrieve_query,
            cfg=cfg,
            deps=deps,
            query_plan=query_plan,
        )
        docs = list(retrieval.documents)
        clarify = retrieval.clarify
    except Exception as exc:  # noqa: BLE001
        retrieval_error = f"{exc.__class__.__name__}: {exc}"
    retrieve_total_ms = (time.perf_counter() - retrieve_start) * 1000.0

    strict_rank = _expected_rank(docs, strict_expected)
    support_rank = _expected_rank(docs, support_expected)
    strict_hit_at_1 = strict_rank == 1
    strict_hit_at_3 = isinstance(strict_rank, int) and strict_rank <= 3
    support_hit_at_1 = support_rank == 1
    support_hit_at_3 = isinstance(support_rank, int) and support_rank <= 3

    query_total_ms = (time.perf_counter() - query_start) * 1000.0
    harness_overhead_ms = max(query_total_ms - retrieve_total_ms, 0.0)
    timeout_exceeded = (
        soft_timeout_seconds > 0
        and query_total_ms > (soft_timeout_seconds * 1000.0)
    )
    stage_timings_ms, stage_call_counts, stage_error_counts = _stage_dict_with_meta(
        stage_timer,
        retrieve_total_ms=retrieve_total_ms,
        harness_overhead_ms=harness_overhead_ms,
    )

    return {
        "run_date": run_date,
        "query_id": query_id,
        "mode": mode,
        "query": query_text,
        "expected_checksums": strict_expected,
        "support_expected_checksums": support_expected,
        "target_profile_when_where": is_targeted_query,
        "top1_checksum": _top_checksum(docs, 0),
        "top2_checksum": _top_checksum(docs, 1),
        "top3_checksum": _top_checksum(docs, 2),
        "top1_score": _top_score(docs, 0),
        "top2_score": _top_score(docs, 1),
        "top3_score": _top_score(docs, 2),
        "hit_at_1": strict_hit_at_1,
        "hit_at_3": strict_hit_at_3,
        "support_hit_at_1": support_hit_at_1,
        "support_hit_at_3": support_hit_at_3,
        "final_context_support_hit": support_hit_at_3,
        "strict_rank": strict_rank,
        "support_rank": support_rank,
        "clarify": clarify,
        "error": retrieval_error,
        "query_duration_ms": round(query_total_ms, 3),
        "soft_timeout_seconds": soft_timeout_seconds,
        "timeout_exceeded": timeout_exceeded,
        "stage_timings_ms": stage_timings_ms,
        "stage_call_counts": stage_call_counts,
        "stage_error_counts": stage_error_counts,
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive_rows = [row for row in rows if row.get("mode") == "positive"]
    control_rows = [row for row in rows if row.get("mode") == "control"]
    targeted_rows = [row for row in positive_rows if bool(row.get("target_profile_when_where"))]

    positive_total = len(positive_rows)
    positive_hit_at_1 = sum(1 for row in positive_rows if bool(row.get("hit_at_1")))
    positive_hit_at_3 = sum(1 for row in positive_rows if bool(row.get("hit_at_3")))
    support_hit_at_1 = sum(1 for row in positive_rows if bool(row.get("support_hit_at_1")))
    support_hit_at_3 = sum(1 for row in positive_rows if bool(row.get("support_hit_at_3")))
    support_context_hits = sum(
        1 for row in positive_rows if bool(row.get("final_context_support_hit"))
    )

    control_total = len(control_rows)
    control_with_results = sum(
        1
        for row in control_rows
        if any(
            isinstance(row.get(field), str) and row.get(field)
            for field in ("top1_checksum", "top2_checksum", "top3_checksum")
        )
    )
    control_clarify_count = sum(
        1 for row in control_rows if isinstance(row.get("clarify"), str) and row.get("clarify")
    )

    targeted_total = len(targeted_rows)
    targeted_support_context_hits = sum(
        1 for row in targeted_rows if bool(row.get("final_context_support_hit"))
    )

    errored_rows = [row for row in rows if isinstance(row.get("error"), str) and row.get("error")]
    timeout_rows = [row for row in rows if bool(row.get("timeout_exceeded"))]

    stage_summary = aggregate_stage_timings(rows)

    return {
        "positive_total": positive_total,
        "positive_hit_at_1": positive_hit_at_1,
        "positive_hit_at_3": positive_hit_at_3,
        "positive_hit_at_1_rate": round((positive_hit_at_1 / positive_total), 4)
        if positive_total
        else 0.0,
        "positive_hit_at_3_rate": round((positive_hit_at_3 / positive_total), 4)
        if positive_total
        else 0.0,
        "positive_support_hit_at_1": support_hit_at_1,
        "positive_support_hit_at_3": support_hit_at_3,
        "positive_support_hit_at_1_rate": round((support_hit_at_1 / positive_total), 4)
        if positive_total
        else 0.0,
        "positive_support_hit_at_3_rate": round((support_hit_at_3 / positive_total), 4)
        if positive_total
        else 0.0,
        "positive_final_context_support_hits": support_context_hits,
        "positive_final_context_support_hit_rate": round(
            (support_context_hits / positive_total), 4
        )
        if positive_total
        else 0.0,
        "control_total": control_total,
        "control_with_results": control_with_results,
        "control_clarify_count": control_clarify_count,
        "target_profile_when_where_total": targeted_total,
        "target_profile_when_where_support_hits": targeted_support_context_hits,
        "target_profile_when_where_support_hit_rate": round(
            (targeted_support_context_hits / targeted_total), 4
        )
        if targeted_total
        else 0.0,
        "target_profile_when_where_query_ids": [
            str(row.get("query_id") or "") for row in targeted_rows
        ],
        "queries_with_errors": len(errored_rows),
        "query_error_ids": [str(row.get("query_id") or "") for row in errored_rows],
        "queries_exceeding_soft_timeout": len(timeout_rows),
        "timeout_query_ids": [str(row.get("query_id") or "") for row in timeout_rows],
        "timing": stage_summary,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        *CSV_BASE_COLUMNS,
        "support_hit_at_1",
        "support_hit_at_3",
        "final_context_support_hit",
        "strict_rank",
        "support_rank",
        "target_profile_when_where",
        "query_duration_ms",
        "timeout_exceeded",
        "stage_timings_ms",
        "stage_call_counts",
        "stage_error_counts",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            normalized["expected_checksums"] = ";".join(row.get("expected_checksums") or [])
            normalized["stage_timings_ms"] = json.dumps(
                row.get("stage_timings_ms") or {},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            normalized["stage_call_counts"] = json.dumps(
                row.get("stage_call_counts") or {},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            normalized["stage_error_counts"] = json.dumps(
                row.get("stage_error_counts") or {},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            writer.writerow(normalized)


def _run_targeted_qa_probe(
    rows: Sequence[Mapping[str, Any]],
    cfg: RetrievalConfig,
) -> dict[str, Any]:
    target_rows = [row for row in rows if bool(row.get("target_profile_when_where"))]
    if not target_rows:
        return {
            "total": 0,
            "answered_without_error": 0,
            "error_count": 0,
            "fallback_count": 0,
            "avg_duration_ms": 0.0,
            "rows": [],
        }

    from qa_pipeline.coordinator import answer_question

    qa_rows: list[dict[str, Any]] = []
    durations: list[float] = []
    answered_without_error = 0
    error_count = 0
    fallback_count = 0

    for row in target_rows:
        query = str(row.get("query") or "")
        start = time.perf_counter()
        context = answer_question(
            query,
            top_k=cfg.top_k,
            retrieval_cfg=cfg,
            temperature=0.0,
            use_cache=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        durations.append(elapsed_ms)

        answer = str(context.answer or "")
        is_error = answer.startswith("[LLM Error:")
        normalized = answer.strip().lower()
        is_fallback = normalized in {
            "",
            "i don't know.",
            "no relevant context found to answer the question.",
            "❌ retrieval failed.",
        }
        if is_error:
            error_count += 1
        elif is_fallback:
            fallback_count += 1
        else:
            answered_without_error += 1

        qa_rows.append(
            {
                "query_id": row.get("query_id"),
                "query": query,
                "duration_ms": round(elapsed_ms, 3),
                "is_error": is_error,
                "is_fallback": is_fallback,
                "answer": answer,
            }
        )

    avg_duration_ms = (sum(durations) / len(durations)) if durations else 0.0
    return {
        "total": len(target_rows),
        "answered_without_error": answered_without_error,
        "error_count": error_count,
        "fallback_count": fallback_count,
        "avg_duration_ms": round(avg_duration_ms, 3),
        "rows": qa_rows,
    }


def run_eval(
    *,
    fixture_path: Path,
    cfg: RetrievalConfig,
    support_labels_path: Optional[Path],
    soft_timeout_seconds: float,
    run_targeted_qa: bool,
) -> dict[str, Any]:
    fixture = _load_json(fixture_path)
    fixture_rows_raw = fixture.get("queries", [])
    fixture_rows: list[dict[str, Any]] = [
        dict(row) for row in fixture_rows_raw if isinstance(row, Mapping)
    ]
    support_overrides = _load_support_overrides(support_labels_path)
    run_date = _iso_date_utc()
    base_reranker = build_configured_reranker() if cfg.enable_rerank else None

    eval_start = time.perf_counter()
    rows = [
        _evaluate_single_query(
            row,
            cfg=cfg,
            run_date=run_date,
            support_overrides=support_overrides,
            soft_timeout_seconds=soft_timeout_seconds,
            base_reranker=base_reranker,
        )
        for row in fixture_rows
    ]
    total_duration_ms = (time.perf_counter() - eval_start) * 1000.0

    summary = _summarize_rows(rows)
    if run_targeted_qa:
        summary["target_profile_when_where_qa"] = _run_targeted_qa_probe(rows, cfg)

    return {
        "generated_at_utc": _utc_now_iso(),
        "fixture": str(fixture_path),
        "config": _cfg_payload(cfg),
        "summary": summary,
        "rows": rows,
        "run_duration_ms": round(total_duration_ms, 3),
    }


def compare_runs(
    *,
    off_run: Mapping[str, Any],
    on_run: Mapping[str, Any],
) -> dict[str, Any]:
    off_summary = off_run.get("summary", {}) if isinstance(off_run, Mapping) else {}
    on_summary = on_run.get("summary", {}) if isinstance(on_run, Mapping) else {}

    off_rows = {
        str(row.get("query_id")): row
        for row in (off_run.get("rows", []) if isinstance(off_run, Mapping) else [])
        if isinstance(row, Mapping) and isinstance(row.get("query_id"), str)
    }
    on_rows = {
        str(row.get("query_id")): row
        for row in (on_run.get("rows", []) if isinstance(on_run, Mapping) else [])
        if isinstance(row, Mapping) and isinstance(row.get("query_id"), str)
    }

    improved_context_support: list[str] = []
    regressed_context_support: list[str] = []
    improved_strict_hit3: list[str] = []
    regressed_strict_hit3: list[str] = []

    for query_id in sorted(set(off_rows) | set(on_rows)):
        off_row = off_rows.get(query_id, {})
        on_row = on_rows.get(query_id, {})
        off_context_hit = bool(off_row.get("final_context_support_hit"))
        on_context_hit = bool(on_row.get("final_context_support_hit"))
        if (not off_context_hit) and on_context_hit:
            improved_context_support.append(query_id)
        if off_context_hit and (not on_context_hit):
            regressed_context_support.append(query_id)

        off_hit3 = bool(off_row.get("hit_at_3"))
        on_hit3 = bool(on_row.get("hit_at_3"))
        if (not off_hit3) and on_hit3:
            improved_strict_hit3.append(query_id)
        if off_hit3 and (not on_hit3):
            regressed_strict_hit3.append(query_id)

    qa_off = off_summary.get("target_profile_when_where_qa", {})
    qa_on = on_summary.get("target_profile_when_where_qa", {})

    return {
        "full_gold_set": {
            "strict_hit_at_1": {
                "off": off_summary.get("positive_hit_at_1", 0),
                "on": on_summary.get("positive_hit_at_1", 0),
                "delta": int(on_summary.get("positive_hit_at_1", 0))
                - int(off_summary.get("positive_hit_at_1", 0)),
            },
            "strict_hit_at_3": {
                "off": off_summary.get("positive_hit_at_3", 0),
                "on": on_summary.get("positive_hit_at_3", 0),
                "delta": int(on_summary.get("positive_hit_at_3", 0))
                - int(off_summary.get("positive_hit_at_3", 0)),
            },
            "final_context_support_hits": {
                "off": off_summary.get("positive_final_context_support_hits", 0),
                "on": on_summary.get("positive_final_context_support_hits", 0),
                "delta": int(on_summary.get("positive_final_context_support_hits", 0))
                - int(off_summary.get("positive_final_context_support_hits", 0)),
            },
        },
        "target_profile_when_where": {
            "query_ids": on_summary.get("target_profile_when_where_query_ids")
            or off_summary.get("target_profile_when_where_query_ids")
            or [],
            "support_hits": {
                "off": off_summary.get("target_profile_when_where_support_hits", 0),
                "on": on_summary.get("target_profile_when_where_support_hits", 0),
                "delta": int(on_summary.get("target_profile_when_where_support_hits", 0))
                - int(off_summary.get("target_profile_when_where_support_hits", 0)),
            },
            "qa_answered_without_error": {
                "off": qa_off.get("answered_without_error", 0)
                if isinstance(qa_off, Mapping)
                else 0,
                "on": qa_on.get("answered_without_error", 0)
                if isinstance(qa_on, Mapping)
                else 0,
                "delta": int(qa_on.get("answered_without_error", 0))
                - int(qa_off.get("answered_without_error", 0))
                if isinstance(qa_off, Mapping) and isinstance(qa_on, Mapping)
                else 0,
            },
        },
        "query_level_deltas": {
            "improved_final_context_support": improved_context_support,
            "regressed_final_context_support": regressed_context_support,
            "improved_strict_hit_at_3": improved_strict_hit3,
            "regressed_strict_hit_at_3": regressed_strict_hit3,
        },
        "timing_dominant_stage_group": {
            "off": (
                off_summary.get("timing", {}).get("dominant_stage_group")
                if isinstance(off_summary.get("timing"), Mapping)
                else None
            ),
            "on": (
                on_summary.get("timing", {}).get("dominant_stage_group")
                if isinstance(on_summary.get("timing"), Mapping)
                else None
            ),
        },
        "run_duration_ms": {
            "off": off_run.get("run_duration_ms", 0),
            "on": on_run.get("run_duration_ms", 0),
            "delta": float(on_run.get("run_duration_ms", 0) or 0.0)
            - float(off_run.get("run_duration_ms", 0) or 0.0),
        },
    }


def _base_cfg(
    *,
    top_k: int,
    top_k_each: int,
    enable_variants: bool,
    enable_query_planning: bool,
    enable_hyde: bool,
    enable_mmr: bool,
    sibling_expansion_enabled: bool,
) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=max(top_k, 1),
        top_k_each=max(top_k_each, 1),
        enable_variants=enable_variants,
        enable_query_planning=enable_query_planning,
        enable_hyde=enable_hyde,
        enable_mmr=enable_mmr,
        enable_rerank=False,
        sibling_expansion_enabled=sibling_expansion_enabled,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic retrieval eval runbooks with timing diagnostics and "
            "sibling-expansion OFF/ON comparison."
        )
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--support-labels", type=Path, default=DEFAULT_SUPPORT_LABELS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-k-each", type=int, default=20)
    parser.add_argument("--enable-variants", action="store_true", default=True)
    parser.add_argument("--disable-variants", action="store_true")
    parser.add_argument("--enable-mmr", action="store_true", default=True)
    parser.add_argument("--disable-mmr", action="store_true")
    parser.add_argument(
        "--soft-query-timeout-seconds",
        type=float,
        default=DEFAULT_SOFT_TIMEOUT_SECONDS,
        help=(
            "Soft timeout threshold for reporting only. Queries are not force-killed; "
            "rows exceeding this duration are flagged in output."
        ),
    )
    parser.add_argument(
        "--sibling-expansion-mode",
        choices=("off", "on", "both"),
        default="both",
        help="Run with sibling expansion disabled, enabled, or both.",
    )
    parser.add_argument(
        "--query-planning-mode",
        choices=("baseline", "planning", "planning_hyde", "compare"),
        default="baseline",
        help=(
            "Run retrieval with baseline rewrites, planning only, planning+HyDE, "
            "or emit all three for ablation comparison."
        ),
    )
    parser.add_argument(
        "--skip-targeted-qa",
        action="store_true",
        help="Skip lightweight end-to-end QA probe for targeted profile when/where subset.",
    )
    return parser.parse_args()


def _derive_output_base(user_output: Optional[Path]) -> Path:
    if user_output is not None:
        return user_output
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return DEFAULT_OUTPUT_DIR / f"retrieval_eval_sibling_expansion_compare_{stamp}.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _run_single_mode(
    *,
    mode: str,
    fixture_path: Path,
    support_labels_path: Optional[Path],
    top_k: int,
    top_k_each: int,
    enable_variants: bool,
    enable_query_planning: bool,
    enable_hyde: bool,
    enable_mmr: bool,
    soft_timeout_seconds: float,
    run_targeted_qa: bool,
) -> dict[str, Any]:
    cfg = _base_cfg(
        top_k=top_k,
        top_k_each=top_k_each,
        enable_variants=enable_variants,
        enable_query_planning=enable_query_planning,
        enable_hyde=enable_hyde,
        enable_mmr=enable_mmr,
        sibling_expansion_enabled=(mode == "on"),
    )
    return run_eval(
        fixture_path=fixture_path,
        cfg=cfg,
        support_labels_path=support_labels_path,
        soft_timeout_seconds=soft_timeout_seconds,
        run_targeted_qa=run_targeted_qa,
    )


def main() -> None:
    args = parse_args()
    output_base = _derive_output_base(args.output)
    fixture_path = args.fixture
    support_labels_path = args.support_labels if args.support_labels else None
    enable_variants = args.enable_variants and not args.disable_variants
    enable_mmr = args.enable_mmr and not args.disable_mmr
    run_targeted_qa = not args.skip_targeted_qa

    modes = ["off", "on"] if args.sibling_expansion_mode == "both" else [args.sibling_expansion_mode]
    if args.query_planning_mode == "compare":
        planning_runs = [
            ("baseline", False, False),
            ("planning", True, False),
            ("planning_hyde", True, True),
        ]
    elif args.query_planning_mode == "planning":
        planning_runs = [("planning", True, False)]
    elif args.query_planning_mode == "planning_hyde":
        planning_runs = [("planning_hyde", True, True)]
    else:
        planning_runs = [("baseline", False, False)]

    mode_artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    soft_timeout = max(float(args.soft_query_timeout_seconds), 0.0)
    for planning_label, enable_query_planning, enable_hyde in planning_runs:
        planning_artifacts: dict[str, dict[str, Any]] = {}
        for mode in modes:
            artifact = _run_single_mode(
                mode=mode,
                fixture_path=fixture_path,
                support_labels_path=support_labels_path,
                top_k=args.top_k,
                top_k_each=args.top_k_each,
                enable_variants=enable_variants,
                enable_query_planning=enable_query_planning,
                enable_hyde=enable_hyde,
                enable_mmr=enable_mmr,
                soft_timeout_seconds=soft_timeout,
                run_targeted_qa=run_targeted_qa,
            )
            planning_artifacts[mode] = artifact

            runbook_json_path = output_base.with_name(
                f"{output_base.stem}_{planning_label}_{mode}.json"
            )
            runbook_csv_path = output_base.with_name(
                f"{output_base.stem}_{planning_label}_{mode}.csv"
            )
            _write_json(runbook_json_path, artifact)
            _write_csv(runbook_csv_path, artifact.get("rows", []))
            print(f"Wrote runbook JSON ({planning_label}/{mode}): {runbook_json_path}")
            print(f"Wrote runbook CSV ({planning_label}/{mode}): {runbook_csv_path}")
        mode_artifacts[planning_label] = planning_artifacts

    # Keep existing sibling on/off comparison when a single planning mode is selected.
    if len(planning_runs) == 1:
        selected_label = planning_runs[0][0]
        selected_artifacts = mode_artifacts.get(selected_label, {})
        if "off" in selected_artifacts and "on" in selected_artifacts:
            comparison = compare_runs(
                off_run=selected_artifacts["off"],
                on_run=selected_artifacts["on"],
            )
            comparison_payload = {
                "generated_at_utc": _utc_now_iso(),
                "fixture": str(fixture_path),
                "support_labels": str(support_labels_path) if support_labels_path else None,
                "mode": "sibling_expansion_off_vs_on",
                "query_planning_mode": selected_label,
                "comparison": comparison,
                "artifacts": {
                    "off_json": str(output_base.with_name(f"{output_base.stem}_{selected_label}_off.json")),
                    "on_json": str(output_base.with_name(f"{output_base.stem}_{selected_label}_on.json")),
                    "off_csv": str(output_base.with_name(f"{output_base.stem}_{selected_label}_off.csv")),
                    "on_csv": str(output_base.with_name(f"{output_base.stem}_{selected_label}_on.csv")),
                },
            }
            _write_json(output_base, comparison_payload)
            print(f"Wrote comparison JSON: {output_base}")

            full_set = comparison.get("full_gold_set", {})
            target = comparison.get("target_profile_when_where", {})
            print(
                "Full gold-set strict hit@3 delta (on-off): "
                f"{full_set.get('strict_hit_at_3', {}).get('delta')}"
            )
            print(
                "Target profile when/where support-hit delta (on-off): "
                f"{target.get('support_hits', {}).get('delta')}"
            )
        return

    # Query planning ablation comparison payload.
    planning_comparisons: dict[str, dict[str, Any]] = {}
    for mode in modes:
        baseline_run = mode_artifacts.get("baseline", {}).get(mode)
        planning_run = mode_artifacts.get("planning", {}).get(mode)
        planning_hyde_run = mode_artifacts.get("planning_hyde", {}).get(mode)
        if baseline_run and planning_run:
            planning_comparisons[f"{mode}_baseline_vs_planning"] = compare_runs(
                off_run=baseline_run,
                on_run=planning_run,
            )
        if planning_run and planning_hyde_run:
            planning_comparisons[f"{mode}_planning_vs_planning_hyde"] = compare_runs(
                off_run=planning_run,
                on_run=planning_hyde_run,
            )

    compare_payload = {
        "generated_at_utc": _utc_now_iso(),
        "fixture": str(fixture_path),
        "support_labels": str(support_labels_path) if support_labels_path else None,
        "mode": "query_planning_compare",
        "sibling_expansion_modes": modes,
        "comparison": planning_comparisons,
        "artifacts": {
            f"{planning_label}_{mode}": {
                "json": str(output_base.with_name(f"{output_base.stem}_{planning_label}_{mode}.json")),
                "csv": str(output_base.with_name(f"{output_base.stem}_{planning_label}_{mode}.csv")),
            }
            for planning_label, _, _ in planning_runs
            for mode in modes
        },
    }
    _write_json(output_base, compare_payload)
    print(f"Wrote query planning comparison JSON: {output_base}")


if __name__ == "__main__":
    main()
