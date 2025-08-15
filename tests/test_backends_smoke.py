from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from config import OPENSEARCH_URL, OPENSEARCH_INDEX, QDRANT_URL, QDRANT_COLLECTION

def test_os_qdrant_up_and_seeded():
    os_client = OpenSearch(hosts=[OPENSEARCH_URL])
    assert os_client.indices.exists(index=OPENSEARCH_INDEX)
    res = os_client.search(index=OPENSEARCH_INDEX, body={"query":{"match_all":{}}})
    assert res["hits"]["total"]["value"] >= 1

    qd = QdrantClient(url=QDRANT_URL)
    info = qd.get_collection(QDRANT_COLLECTION)
    assert info.status is not None
