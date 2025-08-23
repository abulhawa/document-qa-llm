from typing import List

from config import logger
from utils.opensearch_utils import get_chunks_by_paths, set_has_embedding_true_by_ids
from utils import qdrant_utils


def reembed_paths(paths: List[str]) -> int:
    """Recompute embeddings for chunks of the given file paths.

    The chunk texts are pulled from OpenSearch, embedded, and upserted to
    Qdrant. The ``has_embedding`` flag for the processed chunks is then
    flipped to ``True`` in OpenSearch.

    Args:
        paths: List of file paths whose chunks should be re-embedded.

    Returns:
        The number of chunks re-embedded.
    """
    chunks = get_chunks_by_paths(paths)
    if not chunks:
        logger.info("No chunks found for re-embedding.")
        return 0

    qdrant_utils.index_chunks(chunks)

    ids = [c.get("id") for c in chunks if c.get("id")]
    if ids:
        set_has_embedding_true_by_ids(ids)

    logger.info("Re-embedded %d chunk(s) for %d path(s).", len(chunks), len(paths))
    return len(chunks)
