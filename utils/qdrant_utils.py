from qdrant_client import QdrantClient, models
from config import QDRANT_URL, QDRANT_COLLECTION
from typing import Optional

client = QdrantClient(url=QDRANT_URL)


def count_qdrant_chunks_by_checksum(checksum: str) -> Optional[int]:
    """
    Return the number of chunks in Qdrant matching the given checksum.
    """
    try:
        result = client.count(
            collection_name=QDRANT_COLLECTION,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="checksum", match=models.MatchValue(value=checksum)
                    ),
                ]
            ),
            exact=True,
        )
        return result.count
    except Exception as e:
        print(f"❌ Qdrant count error for checksum={checksum}: {e}")
        return None


from typing import Iterable
def delete_vectors_by_checksum(checksum: str) -> None:
    """Delete all vectors in Qdrant for a given checksum."""
    try:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="checksum", match=models.MatchValue(value=checksum)
                        )
                    ]
                )
            ),
        )
    except Exception as e:
        print(f"❌ Qdrant delete error for checksum={checksum}: {e}")


def delete_vectors_many_by_checksum(checksums: Iterable[str]) -> None:
    unique = [c for c in {c for c in checksums if c}]
    if not unique:
        return
    # Qdrant's filter supports OR via 'should'. Chunk to keep payload reasonable.
    CHUNK = 64
    for i in range(0, len(unique), CHUNK):
        part = unique[i:i+CHUNK]
        should_conditions = [
            models.FieldCondition(key="checksum", match=models.MatchValue(value=cs))
            for cs in part
        ]
        try:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(should=should_conditions)
                ),
            )
        except Exception as e:
            print(f"❌ Qdrant batch delete error for {len(part)} checksum(s): {e}")
