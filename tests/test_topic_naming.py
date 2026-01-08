import json
from pathlib import Path
import sys
import types

import pytest


def _install_dependency_stubs() -> None:
    sys.modules.setdefault("requests", types.ModuleType("requests"))

    tracing_stub = types.ModuleType("tracing")

    class DummySpan:
        def set_attribute(self, *_args, **_kwargs):
            return None

    def get_current_span():
        return DummySpan()

    def record_span_error(*_args, **_kwargs):
        return None

    tracing_stub.get_current_span = get_current_span
    tracing_stub.record_span_error = record_span_error
    tracing_stub.OUTPUT_VALUE = "output"
    sys.modules.setdefault("tracing", tracing_stub)

    topic_discovery_stub = types.ModuleType("services.topic_discovery_clusters")

    def load_last_cluster_cache():
        return {}

    topic_discovery_stub.load_last_cluster_cache = load_last_cluster_cache
    sys.modules.setdefault("services.topic_discovery_clusters", topic_discovery_stub)

    opensearch_fulltext_stub = types.ModuleType("utils.opensearch.fulltext")

    def get_fulltext_by_checksum(_checksum):
        return None

    opensearch_fulltext_stub.get_fulltext_by_checksum = get_fulltext_by_checksum
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
