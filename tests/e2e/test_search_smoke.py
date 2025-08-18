import os
import pytest

pytestmark = pytest.mark.e2e

if os.getenv("TEST_MODE", "off") != "e2e":
    pytest.skip("e2e-only smoke test (requires OpenSearch)", allow_module_level=True)

from opensearchpy import OpenSearch, helpers
from config import OPENSEARCH_URL, OPENSEARCH_FULLTEXT_INDEX
from utils.fulltext_search import search_documents


def setup_module(module):
    client = OpenSearch(hosts=[OPENSEARCH_URL])
    if client.indices.exists(index=OPENSEARCH_FULLTEXT_INDEX):
        client.indices.delete(index=OPENSEARCH_FULLTEXT_INDEX)
    client.indices.create(index=OPENSEARCH_FULLTEXT_INDEX)
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
            },
        }
    ]
    helpers.bulk(client, docs)
    client.indices.refresh(index=OPENSEARCH_FULLTEXT_INDEX)


def test_search_documents_smoke():
    res = search_documents("hello")
    assert res["total"] >= 1
