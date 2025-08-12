from functools import lru_cache
from opensearchpy import OpenSearch
from config import OPENSEARCH_URL


@lru_cache(maxsize=1)
def get_client() -> OpenSearch:
    return OpenSearch(hosts=[OPENSEARCH_URL])
