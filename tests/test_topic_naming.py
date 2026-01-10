from dataclasses import replace
import json
from pathlib import Path
import sys
import types

import pytest


def _install_dependency_stubs() -> None:
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    opensearch_stub = types.ModuleType("opensearchpy")

    class OpenSearch:
        def __init__(self, *args, **kwargs):
            return None

    setattr(opensearch_stub, "OpenSearch", OpenSearch)
    sys.modules.setdefault("opensearchpy", opensearch_stub)

    qdrant_stub = types.ModuleType("qdrant_client")
    qdrant_http_stub = types.ModuleType("qdrant_client.http")
    qdrant_models_stub = types.ModuleType("qdrant_client.http.models")

    class QdrantClient:
        def __init__(self, *args, **kwargs):
            return None

    setattr(qdrant_stub, "QdrantClient", QdrantClient)
    setattr(qdrant_http_stub, "models", qdrant_models_stub)
    sys.modules.setdefault("qdrant_client", qdrant_stub)
    sys.modules.setdefault("qdrant_client.http", qdrant_http_stub)
    sys.modules.setdefault("qdrant_client.http.models", qdrant_models_stub)

    tracing_stub = types.ModuleType("tracing")

    class DummySpan:
        def set_attribute(self, *_args, **_kwargs):
            return None

    def get_current_span():
        return DummySpan()

    def record_span_error(*_args, **_kwargs):
        return None

    setattr(tracing_stub, "get_current_span", get_current_span)
    setattr(tracing_stub, "record_span_error", record_span_error)
    setattr(tracing_stub, "OUTPUT_VALUE", "output")
    sys.modules.setdefault("tracing", tracing_stub)

    topic_discovery_stub = types.ModuleType("services.topic_discovery_clusters")

    def load_last_cluster_cache():
        return {}

    setattr(topic_discovery_stub, "load_last_cluster_cache", load_last_cluster_cache)
    sys.modules.setdefault("services.topic_discovery_clusters", topic_discovery_stub)

    opensearch_fulltext_stub = types.ModuleType("utils.opensearch.fulltext")

    def get_fulltext_by_checksum(_checksum):
        return None

    setattr(opensearch_fulltext_stub, "get_fulltext_by_checksum", get_fulltext_by_checksum)
    sys.modules.setdefault("utils.opensearch.fulltext", opensearch_fulltext_stub)


_install_dependency_stubs()

from services import topic_naming


@pytest.fixture()
def cluster_cache_payload() -> dict:
    return {
        "collection": "file_vectors",
        "created_at": "2024-01-01T00:00:00+00:00",
        "params": {
            "min_cluster_size": 2,
            "min_samples": 1,
            "metric": "cosine",
            "use_umap": False,
            "umap": None,
            "macro_grouping": {"min_k": 5, "max_k": 10},
        },
        "vector_count": 2,
        "checksums_hash": "abc123",
        "checksums": ["checksum-a", "checksum-b"],
        "payloads": [
            {
                "checksum": "checksum-a",
                "filename": "Q1-report.pdf",
                "path": "/data/finance/q1-report.pdf",
                "filetype": "pdf",
                "text_full": "Revenue growth and expense review.",
            },
            {
                "checksum": "checksum-b",
                "filename": "Q2-report.pdf",
                "path": "/data/finance/q2-report.pdf",
                "filetype": "pdf",
                "text_full": "Quarterly finance planning overview.",
            },
        ],
        "labels": [1, 1],
        "probs": [0.93, 0.91],
        "clusters": [
            {
                "cluster_id": 1,
                "size": 2,
                "avg_prob": 0.92,
                "centroid": [0.1, 0.2],
                "representative_checksums": ["checksum-a", "checksum-b"],
            }
        ],
    }


@pytest.fixture()
def cluster_profile(cluster_cache_payload: dict) -> topic_naming.ClusterProfile:
    cluster = cluster_cache_payload["clusters"][0]
    representative_files = cluster_cache_payload["payloads"]
    return topic_naming.ClusterProfile(
        cluster_id=cluster["cluster_id"],
        size=cluster["size"],
        avg_prob=cluster["avg_prob"],
        centroid=cluster["centroid"],
        mixedness=0.12,
        representative_checksums=cluster["representative_checksums"],
        representative_files=representative_files,
        representative_paths=[entry["path"] for entry in representative_files],
        representative_snippets=["Revenue growth", "Expense review"],
        keywords=["revenue", "expenses", "planning"],
        top_extensions=[{"extension": ".pdf", "count": 2}],
    )


def test_tokenize_filename() -> None:
    tokens = topic_naming.tokenize_filename("The_Project-Plan_v2.final.pdf")
    assert tokens == ["project", "plan", "v2", "final"]


def test_extract_path_segments_with_root_and_depth() -> None:
    segments = topic_naming.extract_path_segments(
        "/data/projects/acme/2024/report-final.pdf",
        max_depth=2,
        root_path="/data/projects",
    )
    assert segments == ["2024", "report", "final"]


def test_english_only_check_rejects_non_latin_and_german_stopwords() -> None:
    assert not topic_naming.english_only_check("日本語のトピック")
    assert not topic_naming.english_only_check("Und der Plan")
    assert topic_naming.english_only_check("Project Plan")


def test_postprocess_name_title_case_and_word_limit() -> None:
    cleaned = topic_naming.postprocess_name("  a very long topic name with many words  ")
    assert cleaned == "A Very Long Topic Name With"


def test_postprocess_name_trims_and_title_cases() -> None:
    cleaned = topic_naming.postprocess_name("  sales   pipeline   updates  ")
    assert cleaned == "Sales Pipeline Updates"


def test_postprocess_name_caps_length_and_handles_garbage() -> None:
    long_name = "enterprise risk management and compliance overview" * 3
    cleaned = topic_naming.postprocess_name(long_name)
    assert len(cleaned) <= topic_naming._NAME_MAX_CHARS

    assert topic_naming.postprocess_name("misc") == "Untitled"
    assert topic_naming.postprocess_name("!!!") == "Untitled"


def test_cache_key_stability_and_cache_hits(
    cluster_profile: topic_naming.ClusterProfile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(topic_naming, "CACHE_DIR", tmp_path)
    cache_key = topic_naming.hash_profile(cluster_profile, "v1", "test-model")
    assert cache_key == topic_naming.hash_profile(cluster_profile, "v1", "test-model")

    calls = {"count": 0}

    def fake_llm(profile: topic_naming.ClusterProfile) -> topic_naming.NameSuggestion:
        calls["count"] += 1
        payload = json.dumps({"name": "finance overview", "confidence": 0.91})
        parsed = json.loads(payload)
        return topic_naming.NameSuggestion(
            name=parsed["name"],
            confidence=parsed["confidence"],
            source="llm",
        )

    first = topic_naming.suggest_child_name_with_llm(
        cluster_profile,
        model_id="test-model",
        llm_callable=fake_llm,
        allow_cache=True,
    )
    second = topic_naming.suggest_child_name_with_llm(
        cluster_profile,
        model_id="test-model",
        llm_callable=fake_llm,
        allow_cache=True,
    )

    assert calls["count"] == 1
    assert first.name == "finance overview"
    assert second.name == "finance overview"


def test_hash_profile_changes_with_model_prompt_and_profile(
    cluster_profile: topic_naming.ClusterProfile,
) -> None:
    base = topic_naming.hash_profile(cluster_profile, "v1", "model-a")
    assert base == topic_naming.hash_profile(cluster_profile, "v1", "model-a")
    assert base != topic_naming.hash_profile(cluster_profile, "v2", "model-a")
    assert base != topic_naming.hash_profile(cluster_profile, "v1", "model-b")

    updated_profile = replace(
        cluster_profile,
        representative_checksums=cluster_profile.representative_checksums + ["new"],
    )
    assert base != topic_naming.hash_profile(updated_profile, "v1", "model-a")


def test_hash_profile_changes_with_keywords(
    cluster_profile: topic_naming.ClusterProfile,
) -> None:
    base = topic_naming.hash_profile(cluster_profile, "v1", "model-a")
    updated_profile = replace(
        cluster_profile,
        keywords=cluster_profile.keywords + ["forecasting"],
    )
    assert base != topic_naming.hash_profile(updated_profile, "v1", "model-a")


def test_cache_miss_when_disabled(
    cluster_profile: topic_naming.ClusterProfile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(topic_naming, "CACHE_DIR", tmp_path)
    calls = {"count": 0}

    def fake_llm(profile: topic_naming.ClusterProfile) -> topic_naming.NameSuggestion:
        calls["count"] += 1
        payload = json.dumps({"name": "risk review", "confidence": 0.88})
        parsed = json.loads(payload)
        return topic_naming.NameSuggestion(
            name=parsed["name"],
            confidence=parsed["confidence"],
            source="llm",
        )

    first = topic_naming.suggest_child_name_with_llm(
        cluster_profile,
        model_id="test-model",
        llm_callable=fake_llm,
        allow_cache=False,
    )
    second = topic_naming.suggest_child_name_with_llm(
        cluster_profile,
        model_id="test-model",
        llm_callable=fake_llm,
        allow_cache=False,
    )

    assert calls["count"] == 2
    assert first.name == "risk review"
    assert second.name == "risk review"


def test_disambiguate_duplicate_names() -> None:
    names = topic_naming.disambiguate_duplicate_names(
        ["Alpha", "Beta", "Alpha", "Alpha"]
    )
    assert names == ["Alpha", "Beta", "Alpha (2)", "Alpha (3)"]


def test_disambiguate_duplicate_names_with_differentiators() -> None:
    names = topic_naming.disambiguate_duplicate_names(
        ["Alpha", "Alpha", "Alpha"],
        differentiators=["finance", ".pdf", None],
    )
    assert names == ["Alpha", "Alpha (Finance)", "Alpha (Pdf)"]


def test_disambiguate_duplicate_names_collision_fallback() -> None:
    names = topic_naming.disambiguate_duplicate_names(
        ["Alpha", "Alpha", "Alpha"],
        differentiators=["finance", "finance", None],
    )
    assert names == ["Alpha", "Alpha (Finance)", "Alpha (2)"]


def test_mock_llm_uses_deterministic_response(
    cluster_profile: topic_naming.ClusterProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(topic_naming, "check_llm_status", lambda: {"active": True})
    monkeypatch.setattr(topic_naming, "ask_llm", lambda *_args, **_kwargs: "finance review")

    suggestion = topic_naming.suggest_child_name_with_llm(
        cluster_profile,
        model_id="test-model",
        allow_cache=False,
    )

    assert suggestion.name == "Finance Review"


def test_llm_retry_successes_and_fallbacks(
    cluster_profile: topic_naming.ClusterProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(topic_naming, "check_llm_status", lambda: {"active": True})

    responses = iter(["Und der Plan", "Financial Planning"])

    def fake_ask_llm(*_args, **_kwargs) -> str:
        return next(responses)

    monkeypatch.setattr(topic_naming, "ask_llm", fake_ask_llm)

    suggestion = topic_naming.suggest_child_name_with_llm(
        cluster_profile,
        model_id="test-model",
        allow_cache=False,
    )

    assert suggestion.name == "Financial Planning"
    assert suggestion.source == "llm"
    assert suggestion.metadata["llm_cache"]["retry_success"] is True

    retry_responses = iter(["Und der Plan", "Und der Plan"])

    def fake_ask_llm_retry(*_args, **_kwargs) -> str:
        return next(retry_responses)

    monkeypatch.setattr(topic_naming, "ask_llm", fake_ask_llm_retry)

    fallback = topic_naming.suggest_child_name_with_llm(
        cluster_profile,
        model_id="test-model",
        allow_cache=False,
    )

    assert fallback.name == "Revenue Expenses Planning"
    assert fallback.source == "baseline"
    assert fallback.metadata["llm_cache"]["retry_success"] is False
    assert fallback.metadata["llm_cache"]["retry_reason"] == "non_english"


def test_keyword_mixedness_heuristic() -> None:
    assert topic_naming._keyword_mixedness({}, max_keywords=5) == 0.0
    assert topic_naming._keyword_mixedness({"alpha": 10}, max_keywords=5) == 0.0
    balanced = topic_naming._keyword_mixedness({"alpha": 5, "beta": 5}, max_keywords=5)
    assert balanced == pytest.approx(1.0)
    skewed = topic_naming._keyword_mixedness({"alpha": 9, "beta": 1}, max_keywords=5)
    assert 0.0 < skewed < 1.0


def test_significant_terms_short_circuits_local_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubClient:
        def search(self, *, index: str, body: dict) -> dict:
            assert index == topic_naming.FULLTEXT_INDEX
            assert "aggs" in body
            return {
                "aggregations": {
                    "significant_terms": {
                        "buckets": [
                            {"key": "alpha", "score": 3.2},
                            {"key": "the", "score": 5.0},
                            {"key": "beta", "doc_count": 2},
                        ]
                    }
                }
            }

    monkeypatch.setattr(topic_naming, "get_client", lambda: StubClient())
    calls = {"count": 0}

    def fake_fetch_fulltext(_checksum: str) -> dict | None:
        calls["count"] += 1
        return {"path": "/data/file.txt", "filename": "file.txt"}

    monkeypatch.setattr(topic_naming, "_safe_fetch_fulltext", fake_fetch_fulltext)
    keywords = topic_naming.get_significant_keywords_from_os(
        ["checksum-a"],
        snippets=["fallback snippet"],
        max_keywords=5,
    )
    assert keywords[:2] == ["alpha", "beta"]
    assert calls["count"] == 0


def test_select_representative_files_prefers_medoid_and_diversity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = {
        "cluster_id": 1,
        "centroid": [1.0, 0.0],
        "representative_checksums": ["checksum-a", "checksum-b", "checksum-c"],
    }
    checksum_payloads = {
        "checksum-a": {"checksum": "checksum-a", "filename": "a.txt"},
        "checksum-b": {"checksum": "checksum-b", "filename": "b.txt"},
        "checksum-c": {"checksum": "checksum-c", "filename": "c.txt"},
    }

    def fake_load_chunk_embeddings(_checksums: list[str]) -> dict[str, list[dict[str, object]]]:
        return {
            "checksum-a": [{"vector": [1.0, 0.0], "chunk_id": "1"}],
            "checksum-b": [{"vector": [0.99, 0.01], "chunk_id": "2"}],
            "checksum-c": [{"vector": [0.7, 0.7], "chunk_id": "3"}],
        }

    monkeypatch.setattr(topic_naming, "_load_chunk_embeddings", fake_load_chunk_embeddings)
    monkeypatch.setattr(topic_naming, "_safe_fetch_fulltext", lambda _checksum: None)

    representatives = topic_naming.select_representative_files(
        cluster,
        checksum_payloads,
        max_files=2,
    )

    assert [entry["checksum"] for entry in representatives] == [
        "checksum-a",
        "checksum-c",
    ]
