import os
import pytest

# Mark as e2e so unit jobs can exclude it by marker.
pytestmark = pytest.mark.e2e

# Extra safety: if someone runs unit tests without markers locally, skip gracefully.
if os.getenv("TEST_MODE", "off") != "e2e":
    pytest.skip("e2e-only smoke test (requires OpenSearch & Qdrant)", allow_module_level=True)
    
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from config import OPENSEARCH_URL, CHUNKS_INDEX, QDRANT_URL, QDRANT_COLLECTION

def test_os_qdrant_up_and_seeded():
    os_client = OpenSearch(hosts=[OPENSEARCH_URL])
    assert os_client.indices.exists(index=CHUNKS_INDEX)
    res = os_client.search(index=CHUNKS_INDEX, body={"query":{"match_all":{}}})
    assert res["hits"]["total"]["value"] >= 1

    qd = QdrantClient(url=QDRANT_URL)
    info = qd.get_collection(QDRANT_COLLECTION)
    assert info.status is not None
