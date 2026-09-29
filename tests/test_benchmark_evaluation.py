from __future__ import annotations

import math

from scripts.run_benchmark_evaluation import (
    document_id_from_hit,
    document_identity_from_hit,
    query_metrics,
    select_tier_queries,
    summarize_results,
)


def test_tier_selection_is_deterministic_and_nested() -> None:
    rows = [{"query_id": f"q{i}", "text": f"query {i}"} for i in range(20)]
    queries = {"open_ragbench": rows}
    smoke = select_tier_queries(
        queries,
        quotas={"open_ragbench": 5},
        seed=20260928,
    )["open_ragbench"]
    medium = select_tier_queries(
        queries,
        quotas={"open_ragbench": 10},
        seed=20260928,
    )["open_ragbench"]
    repeated = select_tier_queries(
        queries,
        quotas={"open_ragbench": 5},
        seed=20260928,
    )["open_ragbench"]

    assert smoke == repeated
    assert smoke == medium[:5]


def test_query_metrics_support_multiple_and_graded_relevant_documents() -> None:
    metrics = query_metrics(
        ["irrelevant", "gold-b", "gold-a", "gold-b"],
        {"gold-a": 1, "gold-b": 2, "negative": 0},
    )

    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0.0 < metrics["ndcg_at_5"] <= 1.0


def test_document_id_from_benchmark_path() -> None:
    hit = {
        "path": "benchmark://composite-v1/officeqa/1939_03.txt",
        "checksum": "fallback",
    }
    assert document_id_from_hit(hit, benchmark_id="composite-v1") == "1939_03.txt"


def test_document_identity_prefers_content_checksum_for_duplicate_aliases() -> None:
    first = {
        "path": "benchmark://composite-v1/open_ragbench/doc-a",
        "checksum": "same-content-sha",
    }
    duplicate = {
        "path": "benchmark://composite-v1/officeqa/doc-b",
        "checksum": "same-content-sha",
    }

    assert document_identity_from_hit(
        first, benchmark_id="composite-v1"
    ) == document_identity_from_hit(
        duplicate, benchmark_id="composite-v1"
    )


def test_errors_are_counted_as_zero_quality_not_dropped() -> None:
    rows = [
        {
            "track": "open_ragbench",
            "error": None,
            "latency_ms": 10.0,
            "hit_at_1": 1.0,
            "hit_at_3": 1.0,
            "hit_at_5": 1.0,
            "recall_at_1": 1.0,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "mrr": 1.0,
            "ndcg_at_5": 1.0,
        },
        {
            "track": "open_ragbench",
            "error": "backend failure",
            "latency_ms": 30.0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "ndcg_at_5": 0.0,
        },
    ]

    summary = summarize_results(rows)
    track = summary["per_track"]["open_ragbench"]
    assert track["queries"] == 2
    assert track["errors"] == 1
    assert math.isclose(track["recall_at_5"], 0.5)
    assert math.isclose(track["latency_ms"]["mean"], 20.0)
