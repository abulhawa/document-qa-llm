import importlib.util
import pathlib
import sys
import types


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_financial_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("run_financial_eval", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load run_financial_eval.py")
eval_script = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("run_financial_eval", eval_script)
_SPEC.loader.exec_module(eval_script)


def test_financial_eval_fixture_meets_gating_gates():
    payload = eval_script.evaluate_fixture(
        pathlib.Path("tests/fixtures/financial_eval_queries.json")
    )

    assert payload["gates"]["overall_pass"] is True

    baseline = payload["baseline"]["summary"]
    gated = payload["gated"]["summary"]

    assert int(gated["queries_with_suppressed_topk"]) == 0
    assert int(gated["suppressed_docs_topk_total"]) < int(
        baseline["suppressed_docs_topk_total"]
    )
    assert float(gated["avg_preferred_ratio_topk"]) > float(
        baseline["avg_preferred_ratio_topk"]
    )
    assert int(gated["year_leakage_docs_topk_total"]) <= int(
        baseline["year_leakage_docs_topk_total"]
    )
    assert int(gated["fallback_logged_count"]) == int(gated["total_queries"])


def test_load_live_query_rows_filters_tax_finance_area():
    rows = eval_script._load_live_query_rows(
        pathlib.Path("tests/fixtures/retrieval_eval_queries.json"),
        target_areas=["tax_docs", "finance_docs"],
        query_ids=None,
        limit=None,
        top_k=5,
        fallback_budget=2,
    )

    query_ids = {str(row.get("id") or "") for row in rows}
    assert {"Q18", "Q19", "Q20"}.issubset(query_ids)
    assert all(int(row.get("top_k") or 0) == 5 for row in rows)


def test_financial_eval_live_mode_uses_live_retrieval(monkeypatch):
    def _fake_retrieve(query, *, cfg, deps=None, query_plan=None):
        if bool(cfg.financial_enable_gating):
            year = 2023 if "2023" in query else 2022
            docs = [
                {
                    "checksum": "gated-doc",
                    "source_family": "invoice",
                    "document_date": f"{year}-01-01",
                    "mentioned_years": [year],
                    "transaction_dates": [f"{year}-01-10"],
                    "financial_record_type": "expense",
                    "tax_relevance_signals": ["tax"],
                    "text": "invoice tax expense",
                }
            ]
            stage_metadata = {"fallback_used": False, "fallback_stage": None}
            return types.SimpleNamespace(documents=docs, stage_metadata=stage_metadata)

        docs = [
            {
                "checksum": "baseline-doc",
                "source_family": "book",
                "document_date": "2021-01-01",
                "mentioned_years": [2021],
                "transaction_dates": ["2021-01-10"],
                "financial_record_type": "expense",
                "tax_relevance_signals": ["tax"],
                "text": "book tax theory",
            }
        ]
        return types.SimpleNamespace(documents=docs, stage_metadata={})

    monkeypatch.setattr(eval_script, "retrieve", _fake_retrieve)

    payload = eval_script.evaluate_live_corpus(
        pathlib.Path("tests/fixtures/retrieval_eval_queries.json"),
        target_areas=["tax_docs"],
        query_ids=["Q18", "Q19"],
        limit=None,
        top_k=1,
        fallback_budget=1,
    )

    assert payload["mode"] == "live"
    assert payload["gates"]["overall_pass"] is True
    assert payload["query_ids"] == ["Q18", "Q19"]

    baseline = payload["baseline"]["summary"]
    gated = payload["gated"]["summary"]
    assert int(gated["suppressed_docs_topk_total"]) < int(baseline["suppressed_docs_topk_total"])
    assert float(gated["avg_preferred_ratio_topk"]) > float(baseline["avg_preferred_ratio_topk"])
