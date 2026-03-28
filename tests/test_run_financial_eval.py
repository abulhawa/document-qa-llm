import importlib.util
import pathlib
import sys


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
