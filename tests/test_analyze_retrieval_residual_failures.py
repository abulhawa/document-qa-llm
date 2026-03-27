import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace


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


def test_resolve_support_expectations_and_query_mode():
    support_checksums, preferred = analysis_script.resolve_support_expectations(
        strict_expected_checksums=["a", "b"],
        label_override={
            "answer_support_mode": "merge",
            "answer_support_checksums": ["c", "a", " "],
            "benchmark_query_type": analysis_script.QUERY_TYPE_MULTI_SOURCE_FACTUAL,
        },
    )
    assert support_checksums == ["a", "b", "c"]
    assert preferred == "a"

    query_type = analysis_script.resolve_query_benchmark_type(
        fixture_query={},
        label_override={"benchmark_query_type": analysis_script.QUERY_TYPE_MULTI_SOURCE_FACTUAL},
        default_query_type=analysis_script.QUERY_TYPE_CANONICAL_DOCUMENT,
    )
    assert query_type == analysis_script.QUERY_TYPE_MULTI_SOURCE_FACTUAL
    assert (
        analysis_script.benchmark_mode_for_query_type(query_type)
        == analysis_script.BENCHMARK_MODE_ANSWER_SUPPORT
    )

    fallback_type = analysis_script.resolve_query_benchmark_type(
        fixture_query={},
        label_override={},
        default_query_type=analysis_script.QUERY_TYPE_CANONICAL_DOCUMENT,
    )
    assert fallback_type == analysis_script.QUERY_TYPE_CANONICAL_DOCUMENT
    assert (
        analysis_script.benchmark_mode_for_query_type(fallback_type)
        == analysis_script.BENCHMARK_MODE_STRICT_RETRIEVAL
    )


def test_analyze_residual_failures_emits_schema_and_zero_filled_query_type_summary(
    tmp_path,
    monkeypatch,
):
    patha_runbook = {
        "config": {},
        "rows": [
            {
                "mode": "positive",
                "query_id": "Q1",
                "query": "where did ali do his phd",
                "expected_checksums": ["expected-1"],
                "hit_at_1": False,
                "hit_at_3": False,
                "top1_checksum": None,
                "top2_checksum": None,
                "top3_checksum": None,
            }
        ],
    }
    fixture = {
        "queries": [
            {
                "id": "Q1",
                "expected_checksums": ["expected-1"],
            }
        ]
    }
    patha_path = tmp_path / "patha.json"
    fixture_path = tmp_path / "fixture.json"
    output_path = tmp_path / "out.json"
    patha_path.write_text(json.dumps(patha_runbook), encoding="utf-8")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    class _FakeOSClient:
        def get(self, *, index, id):  # noqa: ANN001
            return {
                "_id": id,
                "_source": {
                    "checksum": id,
                    "text_full": "Example text",
                    "path": "doc.txt",
                    "filename": "doc.txt",
                    "doc_type": "cv",
                    "extraction_mode": "native",
                },
            }

        def count(self, *, index, body):  # noqa: ANN001
            return {"count": 1}

    monkeypatch.setattr(analysis_script, "get_client", lambda: _FakeOSClient())
    monkeypatch.setattr(
        analysis_script,
        "retrieve",
        lambda query, cfg: SimpleNamespace(documents=[{"checksum": "expected-1"}]),  # noqa: ARG005
    )
    monkeypatch.setattr(
        analysis_script,
        "count_qdrant_chunks_by_checksum",
        lambda checksum: 1,  # noqa: ARG005
    )

    output = analysis_script.analyze_residual_failures(
        patha_runbook_path=patha_path,
        fixture_path=fixture_path,
        output_path=output_path,
        candidate_depth=5,
        very_low_text_threshold=50,
        support_labels_path=None,
    )
    summary = output["aggregate_summary"]["benchmark_query_type_summary"]
    assert output["schema_version"] == analysis_script.RESIDUAL_FAILURE_ANALYSIS_SCHEMA_VERSION
    assert summary[analysis_script.QUERY_TYPE_CANONICAL_DOCUMENT] == 1
    assert summary[analysis_script.QUERY_TYPE_MULTI_SOURCE_FACTUAL] == 0
    assert summary[analysis_script.QUERY_TYPE_AMBIGUOUS_REVIEWER_NEEDED] == 0
