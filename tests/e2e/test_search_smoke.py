import os
import pytest

pytestmark = pytest.mark.e2e

if os.getenv("TEST_MODE", "off") != "e2e":
    pytest.skip("e2e-only smoke test (requires OpenSearch)", allow_module_level=True)

from opensearchpy import OpenSearch, helpers
from config import OPENSEARCH_URL, OPENSEARCH_FULLTEXT_INDEX
from utils.opensearch_utils import FULLTEXT_INDEX_SETTINGS
from utils.fulltext_search import search_documents


def setup_module(module):
    client = OpenSearch(hosts=[OPENSEARCH_URL], timeout=30)

    # Recreate index with the app's mapping so aggs on filetype/path work
    if client.indices.exists(index=OPENSEARCH_FULLTEXT_INDEX):
        client.indices.delete(index=OPENSEARCH_FULLTEXT_INDEX)
    client.indices.create(index=OPENSEARCH_FULLTEXT_INDEX, body=FULLTEXT_INDEX_SETTINGS)

    # Two small docs in different filetypes
    docs = [
        {
            "_index": OPENSEARCH_FULLTEXT_INDEX,
            "_id": "1",
            "_source": {
                "text_full": "hello world",
                "filename": "file1.txt",
                "path": "/docs/file1.txt",
                "filetype": "txt",
                "modified_at": "2023-01-01",
                "created_at": "2023-01-01",
                "size_bytes": 123,
                "checksum": "chk-1",
            },
        },
        {
            "_index": OPENSEARCH_FULLTEXT_INDEX,
            "_id": "2",
            "_source": {
                "text_full": "another hello planet",
                "filename": "file2.pdf",
                "path": "/docs/file2.pdf",
                "filetype": "pdf",
                "modified_at": "2024-06-01",
                "created_at": "2024-06-01",
                "size_bytes": 456,
                "checksum": "chk-2",
            },
        },
    ]
    helpers.bulk(client, docs)
    client.indices.refresh(index=OPENSEARCH_FULLTEXT_INDEX)


def test_search_documents_smoke():
    # Be explicit about highlight size to avoid default drift
    res = search_documents("hello", fragment_size=200)

    assert res["total"] >= 1
    assert "hits" in res and isinstance(res["hits"], list)
    assert "aggs" in res and "filetypes" in res["aggs"]
    # At least one facet bucket should exist (txt/pdf)
    assert res["aggs"]["filetypes"]["buckets"]
