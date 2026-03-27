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


def test_find_first_answer_support_strict_and_equivalent_gate():
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
    equiv_default = ranking_script.find_first_answer_support(
        docs=docs_equiv,
        expected_checksums=["expected-1"],
        metadata_cache=metadata_cache,
        similarity_cache={},
    )
    assert equiv_default["rank"] is None
    assert equiv_default["support_type"] is None

    equiv_allowed = ranking_script.find_first_answer_support(
        docs=docs_equiv,
        expected_checksums=["expected-1"],
        metadata_cache=metadata_cache,
        similarity_cache={},
        allow_equivalent=True,
    )
    assert equiv_allowed["rank"] == 1
    assert equiv_allowed["support_type"] == "equivalent"
    assert equiv_allowed["matched_expected_checksum"] == "expected-1"


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


def test_load_support_labels_and_query_type_resolution(tmp_path):
    labels = {
        "meta": {
            "default_positive_query_type": ranking_script.QUERY_TYPE_CANONICAL_DOCUMENT
        },
        "overrides": {
            "Q01": {
                "benchmark_query_type": ranking_script.QUERY_TYPE_MULTI_SOURCE_FACTUAL
            }
        },
    }
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels), encoding="utf-8")

    meta, overrides = ranking_script._load_support_labels(path)
    assert meta["default_positive_query_type"] == ranking_script.QUERY_TYPE_CANONICAL_DOCUMENT
    assert "Q01" in overrides

    resolved_multi = ranking_script.resolve_query_benchmark_type(
        fixture_query={},
        label_override=overrides["Q01"],
        default_query_type=ranking_script.QUERY_TYPE_CANONICAL_DOCUMENT,
    )
    assert resolved_multi == ranking_script.QUERY_TYPE_MULTI_SOURCE_FACTUAL
    assert (
        ranking_script.benchmark_mode_for_query_type(resolved_multi)
        == ranking_script.BENCHMARK_MODE_ANSWER_SUPPORT
    )

    resolved_default = ranking_script.resolve_query_benchmark_type(
        fixture_query={},
        label_override={},
        default_query_type=ranking_script.QUERY_TYPE_CANONICAL_DOCUMENT,
    )
    assert resolved_default == ranking_script.QUERY_TYPE_CANONICAL_DOCUMENT
    assert (
        ranking_script.benchmark_mode_for_query_type(resolved_default)
        == ranking_script.BENCHMARK_MODE_STRICT_RETRIEVAL
    )


def test_probe_metrics_by_query_type_breakout():
    rows = [
        {
            "benchmark_query_type": ranking_script.QUERY_TYPE_CANONICAL_DOCUMENT,
            "strict_retrieval_rank_probe": 1,
            "answer_support_rank_probe": 1,
        },
        {
            "benchmark_query_type": ranking_script.QUERY_TYPE_CANONICAL_DOCUMENT,
            "strict_retrieval_rank_probe": 5,
            "answer_support_rank_probe": 5,
        },
        {
            "benchmark_query_type": ranking_script.QUERY_TYPE_MULTI_SOURCE_FACTUAL,
            "strict_retrieval_rank_probe": 4,
            "answer_support_rank_probe": 1,
        },
    ]
    strict = ranking_script.probe_metrics_by_query_type(
        per_query_rows=rows,
        mode=ranking_script.BENCHMARK_MODE_STRICT_RETRIEVAL,
        probe_depth=40,
    )
    support = ranking_script.probe_metrics_by_query_type(
        per_query_rows=rows,
        mode=ranking_script.BENCHMARK_MODE_ANSWER_SUPPORT,
        probe_depth=40,
    )

    canonical_strict = strict[ranking_script.QUERY_TYPE_CANONICAL_DOCUMENT]
    assert canonical_strict["positive_total"] == 2
    assert canonical_strict["hit_at_1_probe"] == 1
    assert canonical_strict["hit_at_3_probe"] == 1
    assert canonical_strict["mrr_probe"] == 0.6

    multi_support = support[ranking_script.QUERY_TYPE_MULTI_SOURCE_FACTUAL]
    assert multi_support["positive_total"] == 1
    assert multi_support["hit_at_1_probe"] == 1
    assert multi_support["hit_at_3_probe"] == 1
    assert multi_support["mrr_probe"] == 1.0

    ambiguous_strict = strict[ranking_script.QUERY_TYPE_AMBIGUOUS_REVIEWER_NEEDED]
    assert ambiguous_strict["positive_total"] == 0
    assert ambiguous_strict["hit_at_1_probe_rate"] == 0.0
    assert ambiguous_strict["mrr_probe"] == 0.0


def test_query_anchor_tokens_filters_stopwords():
    tokens = ranking_script.query_anchor_tokens("Where did I do my PhD in Germany?")
    assert "phd" in tokens
    assert "germany" in tokens
    assert "where" not in tokens
    assert "did" not in tokens
    assert "my" not in tokens


def test_query_conditioned_similarity_candidate_rule():
    expected = (
        "Education: Completed PhD at RWTH Aachen University in Germany in 2022. "
        "Research area: machine learning."
    )
    candidate = (
        "Biography: He completed a PhD at RWTH Aachen University in Germany in 2022, "
        "focused on machine learning systems."
    )
    unrelated = (
        "Biography: He completed a PhD at Oxford University in 2018 focused on economics."
    )

    similarity = ranking_script.query_conditioned_similarity_metrics(
        query="where did i do my phd",
        candidate_text=candidate,
        expected_text=expected,
    )
    assert similarity is not None
    assert ranking_script.is_review_candidate_similarity(similarity) is True

    unrelated_similarity = ranking_script.query_conditioned_similarity_metrics(
        query="where did i do my phd",
        candidate_text=unrelated,
        expected_text=expected,
    )
    assert unrelated_similarity is not None
    assert ranking_script.is_review_candidate_similarity(unrelated_similarity) is False


def test_build_answer_support_review_candidates_returns_suggestions():
    docs = [{"checksum": "cand-1"}, {"checksum": "other-1"}]
    metadata_cache = {
        "exp-1": {
            "text_full": "Completed PhD at RWTH Aachen University in Germany in 2022."
        },
        "cand-1": {
            "text_full": (
                "Profile note: completed a PhD at RWTH Aachen University in Germany in 2022."
            ),
            "path": "candidate.txt",
            "doc_type": "cv",
        },
        "other-1": {"text_full": "Completely unrelated content."},
    }

    candidates = ranking_script.build_answer_support_review_candidates(
        query_text="where did i do my phd",
        docs=docs,
        support_expected_checksums=["exp-1"],
        metadata_cache=metadata_cache,
        similarity_cache={},
        rank_limit=3,
    )
    assert len(candidates) == 1
    assert candidates[0]["checksum"] == "cand-1"
    assert candidates[0]["review_reason"] in {
        "equivalent_fulltext_threshold",
        "query_conditioned_similarity_threshold",
    }


def test_assign_primary_ranking_cause_bucket_rules():
    assert (
        ranking_script.assign_primary_ranking_cause(
            expected_rank_probe=None,
            winner_vector_minus_expected=None,
            winner_lexical_minus_expected=None,
            title_overlap_count_delta=None,
            title_overlap_ratio_delta=None,
            doc_type_prior_delta=None,
            near_duplicate_collision=False,
            chunk_aggregation_bias=False,
        )
        == ranking_script.RANKING_CAUSE_CANDIDATE_GENERATION_MISS
    )
    assert (
        ranking_script.assign_primary_ranking_cause(
            expected_rank_probe=4,
            winner_vector_minus_expected=0.01,
            winner_lexical_minus_expected=0.01,
            title_overlap_count_delta=0,
            title_overlap_ratio_delta=0.0,
            doc_type_prior_delta=0.0,
            near_duplicate_collision=True,
            chunk_aggregation_bias=False,
        )
        == ranking_script.RANKING_CAUSE_SIBLING_COLLISION
    )
    assert (
        ranking_script.assign_primary_ranking_cause(
            expected_rank_probe=4,
            winner_vector_minus_expected=0.01,
            winner_lexical_minus_expected=0.01,
            title_overlap_count_delta=0,
            title_overlap_ratio_delta=0.0,
            doc_type_prior_delta=0.03,
            near_duplicate_collision=False,
            chunk_aggregation_bias=False,
        )
        == ranking_script.RANKING_CAUSE_DOC_TYPE_PRIOR_SUPPRESSION
    )
    assert (
        ranking_script.assign_primary_ranking_cause(
            expected_rank_probe=4,
            winner_vector_minus_expected=0.01,
            winner_lexical_minus_expected=0.01,
            title_overlap_count_delta=-2,
            title_overlap_ratio_delta=-0.4,
            doc_type_prior_delta=0.0,
            near_duplicate_collision=False,
            chunk_aggregation_bias=False,
        )
        == ranking_script.RANKING_CAUSE_TITLE_FILENAME_UNDERWEIGHTING
    )
    assert (
        ranking_script.assign_primary_ranking_cause(
            expected_rank_probe=4,
            winner_vector_minus_expected=0.08,
            winner_lexical_minus_expected=0.0,
            title_overlap_count_delta=0,
            title_overlap_ratio_delta=0.0,
            doc_type_prior_delta=0.0,
            near_duplicate_collision=False,
            chunk_aggregation_bias=False,
        )
        == ranking_script.RANKING_CAUSE_VECTOR_DOMINANCE
    )
    assert (
        ranking_script.assign_primary_ranking_cause(
            expected_rank_probe=4,
            winner_vector_minus_expected=0.01,
            winner_lexical_minus_expected=0.01,
            title_overlap_count_delta=0,
            title_overlap_ratio_delta=0.0,
            doc_type_prior_delta=0.0,
            near_duplicate_collision=False,
            chunk_aggregation_bias=True,
        )
        == ranking_script.RANKING_CAUSE_CHUNK_AGGREGATION_BIAS
    )
    assert (
        ranking_script.assign_primary_ranking_cause(
            expected_rank_probe=4,
            winner_vector_minus_expected=0.01,
            winner_lexical_minus_expected=0.01,
            title_overlap_count_delta=0,
            title_overlap_ratio_delta=0.0,
            doc_type_prior_delta=0.0,
            near_duplicate_collision=False,
            chunk_aggregation_bias=False,
        )
        == ranking_script.RANKING_CAUSE_AMBIGUOUS
    )


def test_build_probe_vs_eval_comparison_reports_discrepancy():
    runbook = {
        "config": {
            "top_k": 3,
            "enable_variants": True,
            "enable_mmr": True,
            "anchored_exact_only": True,
            "anchored_lexical_bias_enabled": True,
            "anchored_fusion_weight_vector": 0.4,
            "anchored_fusion_weight_bm25": 0.6,
        },
        "summary": {
            "positive_total": 2,
            "positive_hit_at_1": 1,
            "positive_hit_at_3": 2,
        },
    }
    archived_rows = [
        {
            "query_id": "Q1",
            "query": "where did ali do his phd",
            "mode": "positive",
            "hit_at_1": True,
            "hit_at_3": True,
            "top1_checksum": "a",
            "top2_checksum": "b",
            "top3_checksum": "c",
        },
        {
            "query_id": "Q2",
            "query": "what is ali latest role",
            "mode": "positive",
            "hit_at_1": False,
            "hit_at_3": True,
            "top1_checksum": "d",
            "top2_checksum": "e",
            "top3_checksum": "f",
        },
    ]
    per_query_rows = [
        {
            "query_id": "Q1",
            "strict_retrieval_hit_at_1_probe": True,
            "strict_retrieval_hit_at_3_probe": True,
            "strict_retrieval_rank_probe": 1,
        },
        {
            "query_id": "Q2",
            "strict_retrieval_hit_at_1_probe": False,
            "strict_retrieval_hit_at_3_probe": False,
            "strict_retrieval_rank_probe": 4,
        },
    ]
    probe_docs = {
        "Q1": [{"checksum": "a"}, {"checksum": "b"}, {"checksum": "c"}],
        "Q2": [{"checksum": "x"}, {"checksum": "y"}, {"checksum": "z"}],
    }
    probe_cfg = ranking_script.RetrievalConfig(
        top_k=40,
        top_k_each=160,
        enable_variants=False,
        enable_mmr=True,
    )

    comparison = ranking_script.build_probe_vs_eval_comparison(
        patha_runbook_path=pathlib.Path("docs/runbooks/patha.json"),
        ranking_artifact_path=pathlib.Path("docs/runbooks/ranking.json"),
        runbook=runbook,
        archived_rows=archived_rows,
        per_query_rows=per_query_rows,
        probe_docs_by_query=probe_docs,
        probe_cfg=probe_cfg,
    )

    strict_metrics = comparison["strict_metric_comparison"]
    assert strict_metrics["archived_eval"]["hit_at_3"] == 2
    assert strict_metrics["deterministic_probe"]["hit_at_3"] == 1
    assert strict_metrics["delta_probe_minus_archived"]["hit_at_3"] == -1
    assert comparison["query_disagreement_summary"]["archived_only_hit_at_3_queries"] == 1
    assert comparison["artifact_method_profiles"][0]["candidate_depth"] == 3
    assert comparison["artifact_method_profiles"][1]["candidate_depth"] == 40
    assert "candidate_depth=3" in comparison["deterministic_explanation"]
