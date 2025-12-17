"""
Compatibility shim forwarding to the new ``ingestion`` package.
Use :func:`ingestion.orchestrator.ingest_one` for new imports.
"""

from ingestion.orchestrator import ingest_one

__all__ = ["ingest_one"]
