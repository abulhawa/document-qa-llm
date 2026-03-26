from __future__ import annotations

import pytest

import utils.opensearch_utils as osu


class _FakeIndices:
    def __init__(self, mapping_by_index):
        self._mapping_by_index = mapping_by_index
        self.put_calls = []

    def get_mapping(self, index):
        mapping = self._mapping_by_index[index]
        return {index: {"mappings": {"properties": mapping}}}

    def put_mapping(self, index, body):
        self.put_calls.append({"index": index, "body": body})


class _FakeClient:
    def __init__(self, mapping_by_index):
        self.indices = _FakeIndices(mapping_by_index)


def test_ensure_identity_metadata_mappings_adds_missing_fields(monkeypatch):
    fake_client = _FakeClient(
        {
            osu.CHUNKS_INDEX: {"text": {"type": "text"}},
            osu.FULLTEXT_INDEX: {"text_full": {"type": "text"}},
        }
    )
    monkeypatch.setattr(osu, "get_client", lambda: fake_client)

    osu.ensure_identity_metadata_mappings()

    assert len(fake_client.indices.put_calls) == 2
    for call in fake_client.indices.put_calls:
        props = call["body"]["properties"]
        assert props["doc_type"]["type"] == "keyword"
        assert props["doc_type_confidence"]["type"] == "float"
        assert props["doc_type_source"]["type"] == "keyword"
        assert props["person_name"]["type"] == "text"
        assert props["authority_rank"]["type"] == "float"


def test_ensure_identity_metadata_mappings_raises_on_incompatible_type(monkeypatch):
    fake_client = _FakeClient(
        {
            osu.CHUNKS_INDEX: {"authority_rank": {"type": "keyword"}},
            osu.FULLTEXT_INDEX: {},
        }
    )
    monkeypatch.setattr(osu, "get_client", lambda: fake_client)

    with pytest.raises(RuntimeError, match="Incompatible mapping detected"):
        osu.ensure_identity_metadata_mappings()

    assert fake_client.indices.put_calls == []
