import importlib.util
import json
import pathlib
import sys


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "investigate_ranking_post_patha.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "investigate_ranking_post_patha",
    _SCRIPT_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load investigate_ranking_post_patha.py")
ranking_script = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("investigate_ranking_post_patha", ranking_script)
_SPEC.loader.exec_module(ranking_script)


def test_reciprocal_rank_and_bucket():
    assert ranking_script.reciprocal_rank(1) == 1.0
    assert ranking_script.reciprocal_rank(4) == 0.25
    assert ranking_script.reciprocal_rank(None) == 0.0

    assert ranking_script.rank_bucket(1, 60) == ranking_script.RANK_BUCKET_1
    assert ranking_script.rank_bucket(3, 60) == ranking_script.RANK_BUCKET_2_3
    assert ranking_script.rank_bucket(9, 60) == ranking_script.RANK_BUCKET_4_10
    assert ranking_script.rank_bucket(40, 60) == ranking_script.RANK_BUCKET_11_TO_DEPTH
    assert (
        ranking_script.rank_bucket(None, 60)
        == ranking_script.RANK_BUCKET_NOT_RETRIEVED
    )


def test_text_similarity_metrics_detects_near_duplicate():
    a = "Ali worked as a data scientist in Berlin in 2024."
    b = "Ali worked as a data scientist in Berlin in 2024."
    metrics = ranking_script.text_similarity_metrics(a, b)
    assert metrics["near_duplicate"] is True
    assert metrics["containment_min"] >= 0.99

    c = "Bitcoin price forecast for tomorrow and market volatility."
    d = "Electric circuits lecture notes about Norton theorem."
    metrics_far = ranking_script.text_similarity_metrics(c, d)
    assert metrics_far["near_duplicate"] is False
    assert metrics_far["containment_min"] < 0.5


def test_auto_answer_likelihood_rule_order():
    assert (
        ranking_script.auto_answer_likelihood(
            hit_at_3=False,
            expected_rank=None,
            near_duplicate=True,
            similarity_containment=0.10,
        )
        == ranking_script.LIKELY_CORRECT
    )
    assert (
        ranking_script.auto_answer_likelihood(
            hit_at_3=True,
            expected_rank=3,
            near_duplicate=False,
            similarity_containment=0.20,
        )
        == ranking_script.POSSIBLY_CORRECT_TOP3
    )
    assert (
        ranking_script.auto_answer_likelihood(
            hit_at_3=False,
            expected_rank=7,
            near_duplicate=False,
            similarity_containment=0.82,
        )
        == ranking_script.POSSIBLY_CORRECT_SIMILAR
    )
    assert (
        ranking_script.auto_answer_likelihood(
            hit_at_3=False,
            expected_rank=None,
            near_duplicate=False,
            similarity_containment=0.20,
        )
        == ranking_script.UNLIKELY_OR_UNKNOWN
    )


def test_is_equivalent_answer_support_thresholds():
    assert (
        ranking_script.is_equivalent_answer_support(
            {
                "near_duplicate": True,
                "containment_min": 0.10,
                "sequence_ratio": 0.10,
                "jaccard": 0.10,
            }
        )
        is True
    )
    assert (
        ranking_script.is_equivalent_answer_support(
            {
                "near_duplicate": False,
                "containment_min": 0.83,
                "sequence_ratio": 0.72,
                "jaccard": 0.30,
            }
        )
        is True
    )
    assert (
        ranking_script.is_equivalent_answer_support(
            {
                "near_duplicate": False,
                "containment_min": 0.70,
                "sequence_ratio": 0.60,
                "jaccard": 0.40,
            }
        )
        is False
    )


def test_find_first_answer_support_strict_and_equivalent():
    similarity_cache = {}
    docs_strict = [
        {"checksum": "expected-1"},
        {"checksum": "other-1"},
    ]
    metadata_cache = {
        "expected-1": {"text_full": "Ali did his PhD in Aachen."},
        "other-1": {"text_full": "Unrelated document text."},
    }
    strict = ranking_script.find_first_answer_support(
        docs=docs_strict,
        expected_checksums=["expected-1"],
        metadata_cache=metadata_cache,
        similarity_cache=similarity_cache,
    )
    assert strict["rank"] == 1
    assert strict["support_type"] == "strict"
    assert strict["checksum"] == "expected-1"

    docs_equiv = [
        {"checksum": "equiv-1"},
        {"checksum": "other-2"},
    ]
    metadata_cache = {
        "expected-1": {"text_full": "Ali did his PhD in Aachen in Germany."},
        "equiv-1": {"text_full": "Ali did his PhD in Aachen in Germany."},
        "other-2": {"text_full": "Completely unrelated content"},
    }
    equiv = ranking_script.find_first_answer_support(
        docs=docs_equiv,
        expected_checksums=["expected-1"],
        metadata_cache=metadata_cache,
        similarity_cache={},
    )
    assert equiv["rank"] == 1
    assert equiv["support_type"] == "equivalent"
    assert equiv["matched_expected_checksum"] == "expected-1"


def test_resolve_support_expectations_merge_and_replace():
    support_merge, preferred_merge = ranking_script.resolve_support_expectations(
        strict_expected_checksums=["a", "b"],
        label_override={
            "answer_support_mode": "merge",
            "answer_support_checksums": ["c", "a", "  "],
            "preferred_checksum": "b",
        },
    )
    assert support_merge == ["a", "b", "c"]
    assert preferred_merge == "b"

    support_replace, preferred_replace = ranking_script.resolve_support_expectations(
        strict_expected_checksums=["a", "b"],
        label_override={
            "answer_support_mode": "replace",
            "answer_support_checksums": ["x", "y", "x"],
        },
    )
    assert support_replace == ["x", "y"]
    assert preferred_replace == "a"


def test_load_support_label_overrides(tmp_path):
    labels = {
        "meta": {"version": 1},
        "overrides": {
            "Q01": {
                "answer_support_mode": "merge",
                "answer_support_checksums": ["chk-1"],
            },
            "Q02": "invalid",
        },
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels), encoding="utf-8")

    overrides = ranking_script._load_support_label_overrides(path)
    assert "Q01" in overrides
    assert "Q02" not in overrides
