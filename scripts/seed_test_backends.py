from datetime import datetime
from opensearchpy import OpenSearch, helpers
from qdrant_client import QdrantClient, models
from config import (
    OPENSEARCH_URL, CHUNKS_INDEX,
    QDRANT_URL, QDRANT_COLLECTION,
    EMBEDDING_SIZE
)

# ---- OpenSearch
os_client = OpenSearch(hosts=[OPENSEARCH_URL])
os_client.indices.create(
    index=CHUNKS_INDEX,
    body={
        "settings": {"index": {"refresh_interval": "1s"}},
        "mappings": {
            "properties": {
                "text": {"type":"text"},
                "path": {"type":"keyword"},
                "modified_at": {"type":"date"},
                "indexed_at": {"type":"date"},
                "checksum": {"type":"keyword"},
                "chunk_index": {"type":"integer"},
                "chunk_char_len": {"type": "integer"},
            }
        }
    }
)

docs = [
    {"_index": CHUNKS_INDEX, "_id": "1", "_op_type": "create",
     "_source": {"text":"Sample sentence about a PhD.",
                 "path":"C:/docs/doc1.txt",
                 "modified_at":"2024-01-01T00:00:00Z",
                 "indexed_at":"2024-01-01T00:00:00Z",
                 "checksum":"chk-1","chunk_index":0,"chunk_char_len":28}},
    {"_index": CHUNKS_INDEX, "_id": "2", "_op_type": "create",
     "_source": {"text":"Another sentence mentioning a city.",
                 "path":"C:/docs/doc2.txt",
                 "modified_at": datetime.now().astimezone().isoformat(),
                 "indexed_at":  datetime.now().astimezone().isoformat(),
                 "checksum":"chk-2","chunk_index":0,"chunk_char_len":35}}
]
helpers.bulk(os_client, docs)
os_client.indices.refresh(index=CHUNKS_INDEX)

# ---- Qdrant
qd = QdrantClient(url=QDRANT_URL)
qd.recreate_collection(
    collection_name=QDRANT_COLLECTION,
    vectors_config=models.VectorParams(size=EMBEDDING_SIZE, distance=models.Distance.COSINE),
)
vec = [0.001 * i for i in range(EMBEDDING_SIZE)]
points = [
    models.PointStruct(id=1, vector=vec,
        payload={"path":"C:/docs/doc1.txt",
                 "page":1,"checksum":"chk-1","chunk_index":0,"modified_at":"2024-01-01T00:00:00Z",
                 "chunk_char_len": 28}),
    models.PointStruct(id=2, vector=vec,
        payload={"path":"C:/docs/doc2.txt",
                 "page":1,"checksum":"chk-2","chunk_index":0,"modified_at": datetime.now().astimezone().isoformat(),
                 "chunk_char_len": 35}),
]
qd.upsert(collection_name=QDRANT_COLLECTION, points=points)
