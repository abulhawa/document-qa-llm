import importlib.util
import pathlib
import sys

from qa_pipeline.types import RetrievedDocument


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_qa_handoff_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("run_qa_handoff_eval", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load run_qa_handoff_eval.py")
eval_script = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("run_qa_handoff_eval", eval_script)
_SPEC.loader.exec_module(eval_script)


def test_pack_docs_by_token_budget_respects_min_chunks():
    docs = [
        RetrievedDocument(text="a" * 400, path="doc-a", checksum="a"),
        RetrievedDocument(text="b" * 400, path="doc-b", checksum="b"),
        RetrievedDocument(text="c" * 400, path="doc-c", checksum="c"),
    ]

    packed, used = eval_script.pack_docs_by_token_budget(
        docs,
        token_budget=120,
        min_chunks=2,
    )

    assert [doc.checksum for doc in packed] == ["a", "b"]
    assert used > 120


def test_pack_docs_by_token_budget_greedily_adds_within_budget():
    docs = [
        RetrievedDocument(text="a" * 200, path="doc-a", checksum="a"),
        RetrievedDocument(text="b" * 200, path="doc-b", checksum="b"),
        RetrievedDocument(text="c" * 200, path="doc-c", checksum="c"),
    ]

    packed, _ = eval_script.pack_docs_by_token_budget(
        docs,
        token_budget=120,
        min_chunks=1,
    )

    assert [doc.checksum for doc in packed] == ["a", "b"]


def test_answer_classification_helpers():
    assert eval_script._is_error_answer("[LLM Error: timeout] service timed out")
    assert eval_script._is_fallback_answer("I don't know.")
    assert not eval_script._is_fallback_answer("The city is Berlin.")


def test_benchmark_integrity_diagnostics_flags_drift_for_profile_query():
    fixture_rows = [
        {
            "id": "Q04",
            "mode": "positive",
            "expected_checksums": ["exp-a", "exp-b"],
            "expected_doc_types": ["cv", "resume"],
        }
    ]

    payloads = {
        "exp-a": {"checksum": "exp-a", "doc_type": "research_paper", "path": "C:/docs/paper.pdf"},
        "exp-b": {"checksum": "exp-b", "doc_type": "invoice", "path": "C:/docs/invoice.pdf"},
    }
    diagnostics = eval_script.build_benchmark_integrity_diagnostics(
        fixture_rows,
        fulltext_fetcher=lambda checksum: payloads.get(checksum),
    )

    summary = diagnostics["summary"]
    q04 = diagnostics["queries"]["Q04"]
    assert summary["drift_count"] == 1
    assert summary["integrity_failure_query_ids"] == ["Q04"]
    assert q04["status"] == "drift"
    assert q04["integrity_failure"] is True
    assert q04["intended_doc_types"] == ["cv", "resume"]
    assert q04["matching_checksums"] == []


def test_summarize_separates_integrity_failures_from_true_retrieval_failures():
    rows = [
        {
            "query_id": "Q04",
            "mode": "positive",
            "answered_without_error": False,
            "support_context_hit": False,
            "strict_context_hit": False,
            "is_error": False,
            "error": None,
            "is_fallback": True,
            "query_duration_ms": 10.0,
            "packing_duration_ms": 1.0,
            "packed_docs_count": 2,
            "packed_context_tokens_est": 200.0,
            "timeout_exceeded": False,
            "benchmark_integrity_failure": True,
            "target_profile_when_where": False,
        },
        {
            "query_id": "Q05",
            "mode": "positive",
            "answered_without_error": True,
            "support_context_hit": True,
            "strict_context_hit": True,
            "is_error": False,
            "error": None,
            "is_fallback": False,
            "query_duration_ms": 20.0,
            "packing_duration_ms": 2.0,
            "packed_docs_count": 3,
            "packed_context_tokens_est": 300.0,
            "timeout_exceeded": False,
            "benchmark_integrity_failure": False,
            "target_profile_when_where": False,
        },
    ]

    summary = eval_script._summarize(rows)
    assert summary["positive_total"] == 2
    assert summary["positive_integrity_failures"] == 1
    assert summary["positive_integrity_failure_query_ids"] == ["Q04"]
    assert summary["positive_integrity_clean_total"] == 1
    assert summary["positive_integrity_clean_support_context_hits"] == 1
    assert summary["positive_true_retrieval_failures_excluding_integrity"] == 0
    assert summary["overall_fallback_rate"] == 0.5
    assert summary["overall_error_rate"] == 0.0
