from __future__ import annotations

import pytest

import utils.opensearch_utils as osu


class _FakeIndices:
    def __init__(self, mapping_by_index, exists_by_index=None):
        self._mapping_by_index = mapping_by_index
        self._exists_by_index = dict(exists_by_index or {})
        self.put_calls = []
        self.create_calls = []

    def get_mapping(self, index):
        mapping = self._mapping_by_index.get(index, {})
        return {index: {"mappings": {"properties": mapping}}}

    def put_mapping(self, index, body):
        self.put_calls.append({"index": index, "body": body})

    def exists(self, index):
        return bool(self._exists_by_index.get(index, False))

    def create(self, index, body, params):
        self.create_calls.append({"index": index, "body": body, "params": params})
        self._exists_by_index[index] = True
        if index not in self._mapping_by_index:
            self._mapping_by_index[index] = (
                body.get("mappings", {}).get("properties", {}) or {}
            )


class _FakeClient:
    def __init__(self, mapping_by_index, exists_by_index=None):
        self.indices = _FakeIndices(mapping_by_index, exists_by_index=exists_by_index)


def test_ensure_financial_metadata_mappings_adds_missing_fields(monkeypatch):
    fake_client = _FakeClient(
        {
            osu.CHUNKS_INDEX: {"text": {"type": "text"}},
            osu.FULLTEXT_INDEX: {"text_full": {"type": "text"}},
        }
    )
    monkeypatch.setattr(osu, "get_client", lambda: fake_client)

    osu.ensure_financial_metadata_mappings()

    assert len(fake_client.indices.put_calls) == 2
    for call in fake_client.indices.put_calls:
        props = call["body"]["properties"]
        assert props["is_financial_document"]["type"] == "boolean"
        assert props["document_date"]["type"] == "date"
        assert props["mentioned_years"]["type"] == "integer"
        assert props["amounts"]["type"] == "double"
        assert props["counterparties"]["type"] == "keyword"
        assert props["financial_metadata_version"]["type"] == "keyword"


def test_ensure_financial_metadata_mappings_raises_on_incompatible_type(monkeypatch):
    fake_client = _FakeClient(
        {
            osu.CHUNKS_INDEX: {"is_financial_document": {"type": "keyword"}},
            osu.FULLTEXT_INDEX: {},
        }
    )
    monkeypatch.setattr(osu, "get_client", lambda: fake_client)

    with pytest.raises(RuntimeError, match="Incompatible mapping detected"):
        osu.ensure_financial_metadata_mappings()

    assert fake_client.indices.put_calls == []


def test_ensure_financial_records_index_creates_when_missing(monkeypatch):
    fake_client = _FakeClient(
        mapping_by_index={},
        exists_by_index={osu.FINANCIAL_RECORDS_INDEX: False},
    )
    monkeypatch.setattr(osu, "get_client", lambda: fake_client)

    osu.ensure_financial_records_index()

    assert len(fake_client.indices.create_calls) == 1
    created = fake_client.indices.create_calls[0]
    assert created["index"] == osu.FINANCIAL_RECORDS_INDEX
    assert created["params"]["wait_for_active_shards"] == "1"
    assert fake_client.indices.put_calls == []


def test_ensure_financial_records_index_adds_missing_fields(monkeypatch):
    fake_client = _FakeClient(
        mapping_by_index={
            osu.FINANCIAL_RECORDS_INDEX: {
                "record_type": {"type": "keyword"},
                "date": {"type": "date"},
            }
        },
        exists_by_index={osu.FINANCIAL_RECORDS_INDEX: True},
    )
    monkeypatch.setattr(osu, "get_client", lambda: fake_client)

    osu.ensure_financial_records_index()

    assert len(fake_client.indices.put_calls) == 1
    props = fake_client.indices.put_calls[0]["body"]["properties"]
    assert props["amount"]["type"] == "double"
    assert props["source_links"]["type"] == "nested"


def test_ensure_financial_records_index_raises_on_incompatible_type(monkeypatch):
    fake_client = _FakeClient(
        mapping_by_index={
            osu.FINANCIAL_RECORDS_INDEX: {
                "record_type": {"type": "keyword"},
                "amount": {"type": "keyword"},
            }
        },
        exists_by_index={osu.FINANCIAL_RECORDS_INDEX: True},
    )
    monkeypatch.setattr(osu, "get_client", lambda: fake_client)

    with pytest.raises(RuntimeError, match="Incompatible mapping detected"):
        osu.ensure_financial_records_index()

    assert fake_client.indices.put_calls == []
