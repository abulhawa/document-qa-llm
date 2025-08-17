"""Recreate the OpenSearch index with parent/child join mapping."""

from utils.opensearch_utils import INDEX_SETTINGS
from core.opensearch_client import get_client
from config import OPENSEARCH_INDEX, logger


def recreate() -> None:
    client = get_client()
    if client.indices.exists(index=OPENSEARCH_INDEX):
        logger.info(f"Deleting existing index {OPENSEARCH_INDEX}")
        client.indices.delete(index=OPENSEARCH_INDEX)

    logger.info(f"Creating index {OPENSEARCH_INDEX} with join mapping")
    client.indices.create(index=OPENSEARCH_INDEX, body=INDEX_SETTINGS)


if __name__ == "__main__":
    recreate()

