import sys
import os
import re
from typing import Dict, List

from core import opensearch_store


def test_search_deduplicates_by_checksum(monkeypatch):
    class DummyClient:
        def search(self, index, body):
            return {
                "hits": {
                    "hits": [
                        {"_id": "1", "_score": 1.0, "_source": {"path": "a", "text": "t1", "chunk_index": 0, "modified_at": "", "checksum": "abc"}},
                        {"_id": "2", "_score": 0.9, "_source": {"path": "b", "text": "t1", "chunk_index": 0, "modified_at": "", "checksum": "abc"}},
                        {"_id": "3", "_score": 0.8, "_source": {"path": "c", "text": "t2", "chunk_index": 0, "modified_at": "", "checksum": "def"}},
                    ]
                }
            }

    monkeypatch.setattr(opensearch_store, "get_client", lambda: DummyClient())
    results = opensearch_store.search("q", top_k=2)
    assert len(results) == 2
    checksums = [r["checksum"] for r in results]
    assert len(set(checksums)) == len(checksums)


def test_search_queries_text_path_and_filename_fields(monkeypatch):
    captured = {}

    class DummyClient:
        def search(self, index, body):
            captured["index"] = index
            captured["body"] = body
            return {"hits": {"hits": []}}

    monkeypatch.setattr(opensearch_store, "get_client", lambda: DummyClient())
    opensearch_store.search("ali cv contact", top_k=3)

    query = captured["body"]["query"]["multi_match"]
    assert query["query"] == "ali cv contact"
    assert query["type"] == "best_fields"
    assert query["operator"] == "or"
    assert query["fields"] == [
        "text^1.0",
        "path^0.35",
        "filename^0.75",
        "filename.keyword^1.10",
    ]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _parse_weighted_field(field_with_boost: str) -> tuple[str, float]:
    if "^" not in field_with_boost:
        return field_with_boost, 1.0
    field, boost = field_with_boost.rsplit("^", 1)
    return field, float(boost)


def _analyzed_field_overlap_score(query: str, value: object, boost: float) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    query_terms = _tokenize(query)
    if not query_terms:
        return 0.0
    overlap = query_terms & _tokenize(value)
    return boost * (len(overlap) / len(query_terms))


def _keyword_exact_match_score(query: str, value: object, boost: float) -> float:
    # filename.keyword is a keyword field; treat it as exact-match only.
    if not isinstance(value, str) or not value.strip():
        return 0.0
    return boost if query.strip().lower() == value.strip().lower() else 0.0


class _BehaviorClient:
    def __init__(self, docs: List[Dict[str, object]]):
        self.docs = docs

    def search(self, index, body):
        query = body["query"]["multi_match"]["query"]
        fields = body["query"]["multi_match"]["fields"]

        hits = []
        for i, src in enumerate(self.docs):
            field_scores = []
            for weighted_field in fields:
                field, boost = _parse_weighted_field(weighted_field)
                if field == "filename.keyword":
                    field_scores.append(
                        _keyword_exact_match_score(query, src.get("filename"), boost)
                    )
                else:
                    field_scores.append(
                        _analyzed_field_overlap_score(query, src.get(field), boost)
                    )

            # multi_match(type=best_fields) is dominated by the best matching field.
            score = max(field_scores) if field_scores else 0.0
            hits.append({"_id": str(i), "_score": score, "_source": src})

        hits.sort(
            key=lambda h: (
                float(h.get("_score", 0.0)),
                str(h.get("_source", {}).get("modified_at", "")),
            ),
            reverse=True,
        )
        return {"hits": {"hits": hits}}


def test_search_filename_exact_name_query_prefers_keyword_exact_match(monkeypatch):
    docs = [
        {
            "checksum": "final",
            "filename": "ali_latest_cv_contact_section.pdf",
            "path": "C:/docs/ali_latest_cv_contact_section.pdf",
            "text": "unrelated content",
            "chunk_index": 0,
            "modified_at": "2026-03-01T00:00:00+00:00",
        },
        {
            "checksum": "draft",
            "filename": "ali_latest_cv_contact_section_draft.pdf",
            "path": "C:/docs/ali_latest_cv_contact_section_draft.pdf",
            "text": "unrelated content",
            "chunk_index": 0,
            "modified_at": "2026-03-02T00:00:00+00:00",
        },
    ]

    monkeypatch.setattr(opensearch_store, "get_client", lambda: _BehaviorClient(docs))
    results = opensearch_store.search("ali_latest_cv_contact_section.pdf", top_k=2)

    assert [r["checksum"] for r in results] == ["final", "draft"]


def test_search_content_query_keeps_text_as_primary_signal(monkeypatch):
    docs = [
        {
            "checksum": "evidence",
            "filename": "meeting_notes.pdf",
            "path": "C:/docs/meeting_notes.pdf",
            "text": "Control approach used in the paper is sliding mode control.",
            "chunk_index": 0,
            "modified_at": "2026-03-01T00:00:00+00:00",
        },
        {
            "checksum": "filename_bait",
            "filename": "control-approach-used-reference-index.pdf",
            "path": "C:/lists/control-approach-used-reference-index.pdf",
            "text": "file inventory listing",
            "chunk_index": 0,
            "modified_at": "2026-03-02T00:00:00+00:00",
        },
    ]

    monkeypatch.setattr(opensearch_store, "get_client", lambda: _BehaviorClient(docs))
    results = opensearch_store.search("what control approach is used", top_k=2)

    assert [r["checksum"] for r in results] == ["evidence", "filename_bait"]


def test_search_does_not_let_artifact_filename_outrank_real_evidence(monkeypatch):
    docs = [
        {
            "checksum": "artifact",
            "filename": "pem-fuel-cell-sliding-mode-control-index.csv",
            "path": "C:/exports/index/pem-fuel-cell-sliding-mode-control-index.csv",
            "text": "file inventory and listing",
            "chunk_index": 0,
            "modified_at": "2026-03-02T00:00:00+00:00",
        },
        {
            "checksum": "evidence",
            "filename": "research-notes.pdf",
            "path": "C:/research/pem/research-notes.pdf",
            "text": "The research paper uses Sliding Mode Control for PEM fuel cells.",
            "chunk_index": 0,
            "modified_at": "2026-03-01T00:00:00+00:00",
        },
    ]

    monkeypatch.setattr(opensearch_store, "get_client", lambda: _BehaviorClient(docs))
    results = opensearch_store.search(
        "In the PEM fuel-cell sliding mode control research paper, what control approach is used?",
        top_k=2,
    )

    assert [r["checksum"] for r in results] == ["evidence", "artifact"]
