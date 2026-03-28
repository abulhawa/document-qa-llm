from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_stubbed_runtime_deps() -> None:
    def _module_available(name: str) -> bool:
        if name in sys.modules:
            return True
        try:
            return importlib.util.find_spec(name) is not None
        except ValueError:
            return True

    tracing = sys.modules.get("tracing")
    if tracing is None:
        tracing = types.ModuleType("tracing")
        sys.modules["tracing"] = tracing

    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def set_attribute(self, *args, **kwargs):
            return None

        def set_status(self, *args, **kwargs):
            return None

        def record_exception(self, *args, **kwargs):
            return None

    if not hasattr(tracing, "start_span"):
        tracing.start_span = lambda *args, **kwargs: _Span()
    if not hasattr(tracing, "record_span_error"):
        tracing.record_span_error = lambda *args, **kwargs: None
    if not hasattr(tracing, "get_current_span"):
        tracing.get_current_span = lambda *args, **kwargs: _Span()
    if not hasattr(tracing, "STATUS_OK"):
        tracing.STATUS_OK = "OK"
    if not hasattr(tracing, "EMBEDDING"):
        tracing.EMBEDDING = "EMBEDDING"
    if not hasattr(tracing, "RETRIEVER"):
        tracing.RETRIEVER = "RETRIEVER"
    if not hasattr(tracing, "INPUT_VALUE"):
        tracing.INPUT_VALUE = "INPUT"
    if not hasattr(tracing, "OUTPUT_VALUE"):
        tracing.OUTPUT_VALUE = "OUTPUT"
    if not hasattr(tracing, "LLM"):
        tracing.LLM = "LLM"
    if not hasattr(tracing, "CHAIN"):
        tracing.CHAIN = "CHAIN"
    if not hasattr(tracing, "TOOL"):
        tracing.TOOL = "TOOL"

    if not _module_available("opensearchpy"):
        opensearch_module = types.ModuleType("opensearchpy")
        opensearch_module.OpenSearch = type("OpenSearch", (), {})
        opensearch_module.RequestsHttpConnection = object
        opensearch_module.exceptions = types.SimpleNamespace(
            OpenSearchException=Exception,
            NotFoundError=Exception,
        )
        sys.modules["opensearchpy"] = opensearch_module

    if not _module_available("qdrant_client"):
        qdrant_module = types.ModuleType("qdrant_client")
        qdrant_module.QdrantClient = type(
            "QdrantClient",
            (),
            {
                "__init__": lambda self, *args, **kwargs: None,
                "get_collections": lambda self: types.SimpleNamespace(collections=[]),
                "create_collection": lambda self, *args, **kwargs: None,
                "search": lambda self, **kwargs: [],
            },
        )
        qdrant_module.models = types.ModuleType("qdrant_client.models")
        sys.modules["qdrant_client"] = qdrant_module

    if "qdrant_client.http.models" not in sys.modules:
        qdrant_http_models = types.ModuleType("qdrant_client.http.models")
        qdrant_http_models.PointStruct = type("PointStruct", (), {})
        qdrant_http_models.PointIdsList = type("PointIdsList", (), {})
        qdrant_http_models.VectorParams = type("VectorParams", (), {})
        distance_module = types.ModuleType("qdrant_client.http.models.Distance")
        distance_module.COSINE = "cosine"
        qdrant_http_models.Distance = distance_module
        sys.modules["qdrant_client.http.models"] = qdrant_http_models

    if not _module_available("numpy"):
        if "core.retrieval.mmr" not in sys.modules:
            mmr_module = types.ModuleType("core.retrieval.mmr")
            mmr_module.mmr_select = lambda query, docs, embed, k=1, lambda_mult=0.5: list(docs)[:k]
            sys.modules["core.retrieval.mmr"] = mmr_module
        if "core.retrieval.dedup" not in sys.modules:
            dedup_module = types.ModuleType("core.retrieval.dedup")
            dedup_module.collapse_near_duplicates = (
                lambda docs, embed_texts, sim_threshold=0.9, keep_limit=64: (list(docs), [])
            )
            sys.modules["core.retrieval.dedup"] = dedup_module


_ensure_stubbed_runtime_deps()

from core.financial_query import detect_financial_query
from core.retrieval.pipeline import retrieve
from core.retrieval.types import QueryPlan, RetrievalConfig, RetrievalDeps


PREFERRED_FAMILIES: Set[str] = {
    "tax_document",
    "bank_statement",
    "receipt",
    "invoice",
    "payment_confirmation",
    "school_fee_letter",
    "official_letter",
}
SUPPRESSED_FAMILIES: Set[str] = {
    "book",
    "course_material",
    "publication",
    "cv",
    "reference",
    "archive_misc",
}

DEFAULT_FIXTURE_PATH = Path("tests/fixtures/financial_eval_queries.json")
DEFAULT_OUTPUT_DIR = Path("docs/runbooks")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_date_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _load_fixture(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _family(doc: Mapping[str, Any]) -> str:
    for key in ("_financial_source_family", "source_family"):
        value = str(doc.get(key) or "").strip().lower()
        if value:
            return value
    doc_type = str(doc.get("doc_type") or "").strip().lower()
    return doc_type or "archive_misc"


def _doc_years(doc: Mapping[str, Any]) -> Set[int]:
    years: Set[int] = set()
    for field in ("mentioned_years", "tax_years_referenced"):
        value = doc.get(field)
        if isinstance(value, list):
            for item in value:
                parsed = _int_or_none(item)
                if parsed is not None:
                    years.add(parsed)
        else:
            parsed = _int_or_none(value)
            if parsed is not None:
                years.add(parsed)

    document_date = str(doc.get("document_date") or "").strip()
    if len(document_date) >= 4 and document_date[:4].isdigit():
        years.add(int(document_date[:4]))

    tx_dates = doc.get("transaction_dates")
    if isinstance(tx_dates, list):
        for item in tx_dates:
            value = str(item or "").strip()
            if len(value) >= 4 and value[:4].isdigit():
                years.add(int(value[:4]))
    return years


def _build_query_plan(row: Mapping[str, Any]) -> QueryPlan:
    query = str(row.get("query") or "")
    detected = detect_financial_query(query)
    target_entity = row.get("target_entity")
    if not isinstance(target_entity, str):
        target_entity = detected.target_entity
    target_concept = row.get("target_concept")
    if not isinstance(target_concept, str):
        target_concept = detected.target_concept
    target_year = _int_or_none(row.get("target_year"))
    if target_year is None:
        target_year = detected.target_year
    return QueryPlan(
        raw_query=query,
        semantic_query=query,
        bm25_query=query,
        clarify=None,
        financial_query_mode=True,
        target_entity=target_entity,
        target_year=target_year,
        target_concept=target_concept,
    )


def _build_cfg(*, top_k: int, fallback_budget: int, financial_enable_gating: bool) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=max(int(top_k), 1),
        top_k_each=max(int(top_k) * 2, 8),
        enable_variants=False,
        enable_mmr=False,
        enable_rerank=False,
        fusion_weight_vector=1.0,
        fusion_weight_bm25=0.0,
        authority_boost_enabled=False,
        recency_boost_enabled=False,
        profile_intent_boost_enabled=False,
        financial_enable_gating=financial_enable_gating,
        financial_fallback_residual_budget=max(int(fallback_budget), 0),
    )


def _build_deps(candidates: Sequence[Mapping[str, Any]]) -> RetrievalDeps:
    docs = [dict(item) for item in candidates if isinstance(item, Mapping)]
    return RetrievalDeps(
        semantic_retriever=lambda query, top_k: docs,
        keyword_retriever=lambda query, top_k: [],
        embed_texts=None,
        cross_encoder=None,
    )


def _run_single(row: Mapping[str, Any], *, financial_enable_gating: bool) -> Dict[str, Any]:
    query_id = str(row.get("id") or "")
    top_k = _int_or_none(row.get("top_k")) or 5
    fallback_budget = _int_or_none(row.get("fallback_residual_budget")) or 2
    plan = _build_query_plan(row)
    cfg = _build_cfg(
        top_k=top_k,
        fallback_budget=fallback_budget,
        financial_enable_gating=financial_enable_gating,
    )
    deps = _build_deps(row.get("candidates") or [])
    output = retrieve(
        str(row.get("query") or ""),
        cfg=cfg,
        deps=deps,
        query_plan=plan,
    )
    docs = list(output.documents)[:top_k]
    families = [_family(doc) for doc in docs]
    preferred_topk = sum(1 for value in families if value in PREFERRED_FAMILIES)
    suppressed_topk = sum(1 for value in families if value in SUPPRESSED_FAMILIES)
    non_preferred_topk = max(len(docs) - preferred_topk, 0)

    target_year = plan.target_year
    cross_year_topk = 0
    if target_year is not None:
        for doc in docs:
            years = _doc_years(doc)
            if years and target_year not in years:
                cross_year_topk += 1

    stage_meta = output.stage_metadata or {}
    fallback_used = bool(stage_meta.get("fallback_used"))
    fallback_logged = ("fallback_used" in stage_meta) and ("fallback_stage" in stage_meta)

    return {
        "query_id": query_id,
        "query": str(row.get("query") or ""),
        "target_year": target_year,
        "top_k": top_k,
        "selected_count": len(docs),
        "selected_checksums": [str(doc.get("checksum") or "") for doc in docs],
        "selected_families": families,
        "preferred_topk": preferred_topk,
        "suppressed_topk": suppressed_topk,
        "non_preferred_topk": non_preferred_topk,
        "preferred_ratio_topk": round(
            float(preferred_topk) / float(max(len(docs), 1)),
            4,
        ),
        "cross_year_topk": cross_year_topk,
        "fallback_used": fallback_used,
        "fallback_logged": fallback_logged,
        "fallback_stage": stage_meta.get("fallback_stage"),
        "stage_metadata": stage_meta,
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    year_scoped = [row for row in rows if isinstance(row.get("target_year"), int)]
    return {
        "total_queries": total,
        "queries_with_suppressed_topk": sum(1 for row in rows if int(row.get("suppressed_topk") or 0) > 0),
        "suppressed_docs_topk_total": sum(int(row.get("suppressed_topk") or 0) for row in rows),
        "queries_with_non_preferred_topk": sum(
            1 for row in rows if int(row.get("non_preferred_topk") or 0) > 0
        ),
        "avg_preferred_ratio_topk": round(
            sum(float(row.get("preferred_ratio_topk") or 0.0) for row in rows)
            / float(max(total, 1)),
            4,
        ),
        "year_scoped_queries": len(year_scoped),
        "year_leakage_queries": sum(
            1 for row in year_scoped if int(row.get("cross_year_topk") or 0) > 0
        ),
        "year_leakage_docs_topk_total": sum(
            int(row.get("cross_year_topk") or 0) for row in year_scoped
        ),
        "fallback_used_count": sum(1 for row in rows if bool(row.get("fallback_used"))),
        "fallback_logged_count": sum(1 for row in rows if bool(row.get("fallback_logged"))),
    }


def _gate_checks(baseline: Mapping[str, Any], gated: Mapping[str, Any]) -> Dict[str, Any]:
    gates = {
        "suppressed_removed_from_topk": int(gated.get("queries_with_suppressed_topk") or 0) == 0,
        "suppressed_reduced_vs_baseline": int(gated.get("suppressed_docs_topk_total") or 0)
        < int(baseline.get("suppressed_docs_topk_total") or 0),
        "preferred_family_dominance_improved": float(gated.get("avg_preferred_ratio_topk") or 0.0)
        > float(baseline.get("avg_preferred_ratio_topk") or 0.0),
        "year_leakage_not_worse": int(gated.get("year_leakage_docs_topk_total") or 0)
        <= int(baseline.get("year_leakage_docs_topk_total") or 0),
        "fallback_logging_complete": int(gated.get("fallback_logged_count") or 0)
        == int(gated.get("total_queries") or 0),
    }
    return {
        "overall_pass": all(gates.values()),
        "checks": gates,
    }


def evaluate_fixture(fixture_path: Path) -> Dict[str, Any]:
    fixture = _load_fixture(fixture_path)
    query_rows = [
        dict(item)
        for item in (fixture.get("queries") or [])
        if isinstance(item, Mapping)
    ]

    baseline_rows = [_run_single(row, financial_enable_gating=False) for row in query_rows]
    gated_rows = [_run_single(row, financial_enable_gating=True) for row in query_rows]

    baseline_summary = _summarize(baseline_rows)
    gated_summary = _summarize(gated_rows)
    gate_results = _gate_checks(baseline_summary, gated_summary)

    return {
        "generated_at_utc": _utc_now_iso(),
        "fixture": str(fixture_path),
        "baseline": {
            "config": {"financial_enable_gating": False},
            "summary": baseline_summary,
            "rows": baseline_rows,
        },
        "gated": {
            "config": {"financial_enable_gating": True},
            "summary": gated_summary,
            "rows": gated_rows,
        },
        "deltas": {
            "suppressed_docs_topk_total_delta": int(gated_summary["suppressed_docs_topk_total"])
            - int(baseline_summary["suppressed_docs_topk_total"]),
            "avg_preferred_ratio_topk_delta": round(
                float(gated_summary["avg_preferred_ratio_topk"])
                - float(baseline_summary["avg_preferred_ratio_topk"]),
                4,
            ),
            "year_leakage_docs_topk_total_delta": int(gated_summary["year_leakage_docs_topk_total"])
            - int(baseline_summary["year_leakage_docs_topk_total"]),
        },
        "gates": gate_results,
    }


def _default_output_path() -> Path:
    return DEFAULT_OUTPUT_DIR / f"financial_eval_{_iso_date_utc()}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finance retrieval benchmark (baseline vs finance-gated)."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--output", type=Path, default=_default_output_path())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_fixture(args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Wrote finance eval: {args.output}")
    print(
        "Gate checks: "
        + ", ".join(
            f"{name}={'PASS' if ok else 'FAIL'}"
            for name, ok in payload["gates"]["checks"].items()
        )
    )


if __name__ == "__main__":
    main()
