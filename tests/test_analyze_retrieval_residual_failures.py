import importlib.util
import pathlib
import sys


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_retrieval_residual_failures.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "analyze_retrieval_residual_failures",
    _SCRIPT_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load analyze_retrieval_residual_failures.py")
analysis_script = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("analyze_retrieval_residual_failures", analysis_script)
_SPEC.loader.exec_module(analysis_script)


def test_classify_query_anchor_levels():
    assert (
        analysis_script.classify_query_anchor(
            "In Ali's latest CV, what is his most recent job title?"
        )
        == "anchored"
    )
    assert analysis_script.classify_query_anchor("what is he cv about") == "semi-anchored"
    assert analysis_script.classify_query_anchor("tell me more about it") == "unanchored"


def test_assign_primary_bucket_ordering():
    assert (
        analysis_script.assign_primary_bucket(
            relevant_in_top3=True,
            relevant_in_candidates=True,
            expected_rank_in_candidates=2,
            text_gap_evidence=False,
            corpus_absence_evidence=False,
            text_available_evidence=True,
        )
        == analysis_script.BUCKET_TOP3_NOT_TOP1
    )
    assert (
        analysis_script.assign_primary_bucket(
            relevant_in_top3=False,
            relevant_in_candidates=True,
            expected_rank_in_candidates=9,
            text_gap_evidence=False,
            corpus_absence_evidence=False,
            text_available_evidence=True,
        )
        == analysis_script.BUCKET_RETRIEVED_BELOW_TOP3
    )
    assert (
        analysis_script.assign_primary_bucket(
            relevant_in_top3=False,
            relevant_in_candidates=False,
            expected_rank_in_candidates=None,
            text_gap_evidence=False,
            corpus_absence_evidence=True,
            text_available_evidence=False,
        )
        == analysis_script.BUCKET_CORPUS_ABSENCE
    )


def test_build_ocr_recommendation_thresholds():
    yes_rec = analysis_script.build_ocr_recommendation(
        {analysis_script.BUCKET_TEXT_GAP: 4, analysis_script.BUCKET_AMBIGUOUS: 2},
        total_failed=10,
    )
    assert yes_rec["decision"] == "YES"
    assert yes_rec["recommend_ocr_canary_now"] is True

    no_rec = analysis_script.build_ocr_recommendation(
        {
            analysis_script.BUCKET_TEXT_GAP: 2,
            analysis_script.BUCKET_RETRIEVED_BELOW_TOP3: 5,
        },
        total_failed=10,
    )
    assert no_rec["decision"] == "NO"
    assert no_rec["recommend_ocr_canary_now"] is False
    assert (
        no_rec["evidence"]["dominant_bucket"]
        == analysis_script.BUCKET_RETRIEVED_BELOW_TOP3
    )
