"""Use case for topic discovery review analytics."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

GENERIC_NAME_RE = re.compile(
    r"^(Misc|Other|Documents|Files|Review)(?:\b|\s|\u2014|-)",
    re.I,
)
FALLBACK_REASONS = {
    "llm_timeout",
    "llm_invalid_json",
    "llm_unreachable",
    "llm_model_not_loaded",
    "cache_hit_baseline",
}


def normalize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        level = _first_present(row, ["level", "cluster_level", "type"], "n/a")
        identifier = _first_present(row, ["id", "cluster_id", "parent_id"], "n/a")
        proposed_name = _first_present(
            row, ["proposed_name", "name", "topic_name"], "n/a"
        )
        parent_name = _first_present(row, ["parent_name", "parent"], "")
        confidence = _first_present(row, ["confidence", "score", "name_confidence"], None)
        mixedness = _first_present(row, ["mixedness", "cluster_mixedness"], None)
        warnings = _ensure_list(_first_present(row, ["warnings", "warning"], []))
        rationale = _first_present(row, ["rationale", "explanation"], "")
        llm_used = bool(_first_present(row, ["llm_used"], False))
        cache_hit = bool(_first_present(row, ["cache_hit"], False))
        cache_bypassed = bool(_first_present(row, ["cache_bypassed"], False))
        fallback_reason = _first_present(row, ["fallback_reason"], "")
        error_summary = _first_present(row, ["error_summary"], "")
        keywords = _ensure_list(
            _first_present(row, ["keywords", "significant_terms", "terms"], [])
        )
        example_paths = _ensure_list(
            _first_present(
                row,
                ["example_paths", "representative_paths", "paths"],
                [],
            )
        )
        snippets = _ensure_list(
            _first_present(row, ["snippets", "representative_snippets"], [])
        )

        normalized.append(
            {
                "id": identifier,
                "level": level,
                "proposed_name": proposed_name,
                "parent_name": parent_name,
                "confidence": confidence,
                "mixedness": mixedness,
                "warnings": warnings,
                "rationale": rationale,
                "llm_used": llm_used,
                "cache_hit": cache_hit,
                "cache_bypassed": cache_bypassed,
                "fallback_reason": fallback_reason,
                "error_summary": error_summary,
                "keywords": keywords,
                "example_paths": example_paths,
                "snippets": snippets,
            }
        )
    return normalized


def add_status_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["status"] = df.apply(_status_category, axis=1)
    df["row_key"] = df.apply(lambda row: f"{row['level']}:{row['id']}", axis=1)
    df["warnings_text"] = df["warnings"].apply(_warnings_text)
    return df


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    df["needs_review"] = df.apply(
        lambda row: needs_review(
            row, filters["mixedness_threshold"], filters["confidence_threshold"]
        ),
        axis=1,
    )

    if filters["selected_levels"]:
        df = df[df["level"].isin(filters["selected_levels"])]
    if filters["selected_status"]:
        df = df[df["status"].isin(filters["selected_status"])]

    df = apply_range_filters(
        df,
        mixedness_range=filters["mixedness_range"],
        confidence_range=filters["confidence_range"],
    )

    if filters["needs_review_only"]:
        df = df[df["needs_review"]]

    if filters["search_query"]:
        query = filters["search_query"].lower()
        search_mask = df["proposed_name"].fillna("").str.lower().str.contains(query)
        example_paths = df["example_paths"].apply(lambda paths: " ".join(paths).lower())
        search_mask |= example_paths.str.contains(query)
        df = df[search_mask]

    return df


def apply_range_filters(
    df: pd.DataFrame,
    *,
    mixedness_range: tuple[float, float],
    confidence_range: tuple[float, float],
) -> pd.DataFrame:
    def _range_filter(series: pd.Series, min_value: float, max_value: float) -> pd.Series:
        if min_value == 0.0 and max_value == 1.0:
            return pd.Series([True] * series.size, index=series.index)
        return series.between(min_value, max_value, inclusive="both")

    if "mixedness" in df:
        mixedness_mask = _range_filter(
            df["mixedness"].fillna(-1.0),
            mixedness_range[0],
            mixedness_range[1],
        )
        if mixedness_range != (0.0, 1.0):
            mixedness_mask &= df["mixedness"].notna()
        df = df[mixedness_mask]

    if "confidence" in df:
        confidence_mask = _range_filter(
            df["confidence"].fillna(-1.0),
            confidence_range[0],
            confidence_range[1],
        )
        if confidence_range != (0.0, 1.0):
            confidence_mask &= df["confidence"].notna()
        df = df[confidence_mask]

    return df


def build_metrics(
    df: pd.DataFrame, filtered_df: pd.DataFrame, filters: dict[str, Any]
) -> dict[str, Any]:
    metric_source = filtered_df if not filtered_df.empty else df
    total_items = len(metric_source)
    llm_used_pct = metric_source["llm_used"].mean() * 100 if total_items else 0.0
    cache_hit_pct = metric_source["cache_hit"].mean() * 100 if total_items else 0.0
    baseline_pct = (
        (
            ((~metric_source["llm_used"]) & (~metric_source["cache_hit"]))
            | metric_source["fallback_reason"].fillna("").astype(bool)
        ).mean()
        * 100
        if total_items
        else 0.0
    )
    high_mixedness_pct = (
        (metric_source["mixedness"].fillna(-1.0) >= filters["mixedness_threshold"]).mean()
        * 100
        if total_items
        else 0.0
    )
    median_confidence = metric_source["confidence"].median() if total_items else None
    p10_confidence = metric_source["confidence"].quantile(0.1) if total_items else None
    needs_review_count = int(metric_source["needs_review"].sum())

    return {
        "total_items": total_items,
        "llm_used_pct": llm_used_pct,
        "cache_hit_pct": cache_hit_pct,
        "baseline_pct": baseline_pct,
        "high_mixedness_pct": high_mixedness_pct,
        "median_confidence": median_confidence,
        "p10_confidence": p10_confidence,
        "needs_review_count": needs_review_count,
    }


def build_snapshot_payload(filtered_df: pd.DataFrame) -> str:
    filtered_rows = filtered_df.drop(columns=["warnings_text"]).to_dict(orient="records")
    return json.dumps(filtered_rows, indent=2, default=str)


def save_snapshot(payload: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = Path(".cache") / "topic_naming" / "review_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"review_snapshot_{timestamp}.json"
    snapshot_path.write_text(payload, encoding="utf-8")
    return snapshot_path


def needs_review(
    row: dict[str, Any],
    mixedness_threshold: float,
    confidence_threshold: float,
) -> bool:
    mixedness = row.get("mixedness")
    confidence = row.get("confidence")
    warnings_text = _warnings_text(row.get("warnings", []))
    proposed_name = str(row.get("proposed_name") or "")
    fallback_reason = str(row.get("fallback_reason") or "")
    mixedness_flag = mixedness is not None and mixedness >= mixedness_threshold
    confidence_flag = confidence is not None and confidence <= confidence_threshold
    warning_flag = bool(re.search(r"review|mixed", warnings_text, re.I))
    name_flag = bool(GENERIC_NAME_RE.search(proposed_name))
    fallback_flag = fallback_reason in FALLBACK_REASONS
    return bool(
        mixedness_flag or confidence_flag or warning_flag or name_flag or fallback_flag
    )


def warnings_text(warnings: list[str]) -> str:
    return _warnings_text(warnings)


def is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(entry) for entry in value]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _status_category(row: dict[str, Any]) -> str:
    if row.get("llm_used"):
        return "LLM"
    if row.get("cache_hit"):
        return "Cache"
    return "Baseline"


def _warnings_text(warnings: list[str]) -> str:
    return "; ".join([entry for entry in warnings if entry])
