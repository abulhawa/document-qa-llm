import importlib.util
import pathlib
import sys


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_retrieval_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("run_retrieval_eval", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load run_retrieval_eval.py")
eval_script = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("run_retrieval_eval", eval_script)
_SPEC.loader.exec_module(eval_script)


def test_resolve_support_checksums_merge_and_replace():
    merged = eval_script.resolve_support_checksums(
        strict_expected_checksums=["a", "b"],
        label_override={
            "answer_support_mode": "merge",
            "answer_support_checksums": ["c", "a", " "],
        },
    )
    assert merged == ["a", "b", "c"]

    replaced = eval_script.resolve_support_checksums(
        strict_expected_checksums=["a", "b"],
        label_override={
            "answer_support_mode": "replace",
            "answer_support_checksums": ["x", "y", "x"],
        },
    )
    assert replaced == ["x", "y"]


def test_is_profile_when_where_query():
    assert eval_script.is_profile_when_where_query(
        {
            "mode": "positive",
            "query": "Where did Ali do his PhD studies?",
            "target_areas": ["career_cv_docs"],
            "expected_doc_types": ["cv"],
            "notes": "Profile and education fact.",
        }
    )
    assert not eval_script.is_profile_when_where_query(
        {
            "mode": "positive",
            "query": "From Ali's latest CV, list the main technical skills.",
            "target_areas": ["career_cv_docs"],
            "expected_doc_types": ["cv"],
            "notes": "Profile intent but not when/where phrasing.",
        }
    )


def test_aggregate_stage_timings_identifies_dominant_group():
    rows = [
        {
            "stage_timings_ms": {
                "semantic_retriever": 120.0,
                "keyword_retriever": 30.0,
                "sibling_fetch": 5.0,
                "harness_overhead": 8.0,
            }
        },
        {
            "stage_timings_ms": {
                "semantic_retriever": 80.0,
                "keyword_retriever": 20.0,
                "sibling_fetch": 6.0,
                "harness_overhead": 4.0,
            }
        },
    ]
    summary = eval_script.aggregate_stage_timings(rows)
    groups = summary["stage_group_totals_ms"]
    assert groups["live_index_queries"] == 250.0
    assert groups["context_expansion"] == 11.0
    assert summary["dominant_stage_group"] == "live_index_queries"
