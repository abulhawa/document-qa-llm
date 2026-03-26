# Chunk Size Migration Runbook (P5)

This runbook covers the controlled migration from chunking `400/50` to `800/100`.

## Scope

- Runtime default chunking changes:
  - `CHUNK_SIZE=800`
  - `CHUNK_OVERLAP=100`
- Applies only to newly ingested or re-ingested files.
- Existing indexed content keeps old chunking until re-ingested.

## Preconditions

- P0-P3 are deployed and validated.
- OpenSearch, Qdrant, worker, and app are healthy.
- A rollback window is available.

## Safety Notes

- Mixed old/new chunking in the same index can affect retrieval consistency.
- Re-ingestion can increase temporary index/storage load.
- Use batch/prefix rollouts and verify each batch before continuing.

## Migration Steps

1. Set environment:

```env
CHUNK_SIZE=800
CHUNK_OVERLAP=100
```

2. Restart app + worker so the new config is active.

3. Pick a small prefix for canary rollout first.

4. (Optional but recommended) Preview destructive purge for that prefix:

```powershell
python scripts/purge_by_prefix.py "C:/your/prefix" --dry-run
```

5. Purge old indexed data for the same prefix:

```powershell
python scripts/purge_by_prefix.py "C:/your/prefix"
```

6. Re-ingest the same prefix from Streamlit:
   - Use the Ingest / Index Viewer / Watchlist re-ingest flows for that prefix.

7. Validate retrieval quality on the manual QA set, then repeat by prefix batches.

## Validation Checklist

- Chunk docs are present again for migrated files.
- Source quality remains stable or improves on the QA set.
- Duplicate-source rate does not regress.
- No sustained ingestion errors in logs.

## Rollback

1. Revert chunk env values back to `400/50`.
2. Restart app + worker.
3. Re-ingest affected prefixes again if you need complete chunk-shape rollback.
