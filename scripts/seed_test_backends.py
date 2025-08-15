from datetime import datetime, timezone
from opensearchpy import OpenSearch, helpers
from qdrant_client import QdrantClient, models
from config import (
    OPENSEARCH_URL, OPENSEARCH_INDEX,
    QDRANT_URL, QDRANT_COLLECTION,
    EMBEDDING_SIZE
)

# ---- OpenSearch
os_client = OpenSearch(hosts=[OPENSEARCH_URL])
os_client.indices.create(
    index=OPENSEARCH_INDEX,
    body={
        "settings": {"index": {"refresh_interval": "1s"}},
        "mappings": {
            "properties": {
                "text": {"type":"text"},
                "path": {"type":"keyword"},
                "modified_at": {"type":"date"},
                "indexed_at": {"type":"date"},
                "checksum": {"type":"keyword"},
                "chunk_index": {"type":"integer"}
            }
        }
    }
)

docs = [
    {"_index": OPENSEARCH_INDEX, "_id": "1",
     "_source": {"text":"Sample sentence about a PhD.",
                 "path":"C:/docs/doc1.txt",
                 "modified_at":"2024-01-01T00:00:00Z",
                 "indexed_at":"2024-01-01T00:00:00Z",
                 "checksum":"chk-1","chunk_index":0}},
    {"_index": OPENSEARCH_INDEX, "_id": "2",
     "_source": {"text":"Another sentence mentioning a city.",
                 "path":"C:/docs/doc2.txt",
                 "modified_at": datetime.now(timezone.utc).isoformat(),
                 "indexed_at":  datetime.now(timezone.utc).isoformat(),
                 "checksum":"chk-2","chunk_index":0}}
]
helpers.bulk(os_client, docs)
os_client.indices.refresh(index=OPENSEARCH_INDEX)

# ---- Qdrant
qd = QdrantClient(url=QDRANT_URL)
qd.recreate_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=models.VectorParams(size=EMBEDDING_SIZE, distance=models.Distance.COSINE),
)
vec = [0.001 * i for i in range(EMBEDDING_SIZE)]
points = [
    models.PointStruct(id=1, vector=vec,
        payload={"path":"C:/docs/doc1.txt","text":"Sample sentence about a PhD.",
                 "page":1,"checksum":"chk-1","chunk_index":0,"modified_at":"2024-01-01T00:00:00Z"}),
    models.PointStruct(id=2, vector=vec,
        payload={"path":"C:/docs/doc2.txt","text":"Another sentence mentioning a city.",
                 "page":1,"checksum":"chk-2","chunk_index":0,"modified_at": datetime.now(timezone.utc).isoformat()}),
]
qd.upsert(collection_name=QDRANT_COLLECTION, points=points)
