# File Path Re-Sync

File Path Re-Sync reconciles files on disk with the document metadata stored in
the search indexes. It is for path drift: files moved, copied, deleted, or
replaced after ingestion.

## System Role

The checksum is the content identity. File paths are metadata.

- `full_text` stores the canonical `path`, `filename`, `checksum`, and optional
  `aliases`.
- chunk documents store `path`, `filename`, and `checksum` for retrieval and
  citations.
- Qdrant payloads also store `path`, `filename`, and `checksum`.

When a canonical path changes, all three stores must stay aligned or retrieval
can return stale citations.

## Pipeline

1. `pages/7_file_resync.py` collects roots, options, and displays the plan.
2. `app/usecases/file_resync_usecase.py` converts UI request/response schemas.
3. `core/sync/file_resync.py` scans disk, loads the index snapshot, builds the
   reconciliation plan, and applies approved actions.

The scan phase is a dry run. It does not modify indexes.

## Scan Filters

The page exposes scan filters before the plan is built:

- allowed file extensions
- minimum file size in bytes
- temporary filename prefixes
- temporary filename suffixes
- ignored directory names

These filters run before checksum calculation. Raising the minimum size or
ignoring temp folders reduces scan cost and avoids indexing partial files.
Lowering the size threshold or clearing temp patterns can be useful for tests or
for intentionally indexing small text files.

## Buckets

- `SAFE`: unambiguous metadata updates, such as adding an alias or changing a
  missing canonical path when exactly one disk path exists for that checksum.
- `INFO`: files found on disk but not indexed. They can be ingested when
  `ingest_missing` is enabled.
- `REVIEW`: cases needing human judgment, such as replaced paths, orphan
  candidates, or multiple possible canonical paths.
- `BLOCKED`: unsafe index states, such as duplicate full-text documents for the
  same checksum. These should be repaired before applying a resync plan.

## Actions

- `INGEST_NEW`: ingest a disk file whose checksum is not in the index.
- `ADD_ALIAS`: add a new disk path as an alias for existing content.
- `REMOVE_ALIAS`: remove an alias that is missing within scanned roots.
- `SET_CANONICAL`: update the canonical path and propagate it to chunks and
  Qdrant payloads.
- `DELETE_CONTENT`: delete vectors, chunks, and full-text content for a checksum.
  This only runs when the matching destructive option is enabled at apply time.

## Safety Rules

- A partial root scan must not prove content is orphaned unless every indexed
  path for that checksum was inside the scanned roots.
- Canonical path changes are automatic only when there is one disk path for the
  checksum. Multiple disk paths are `REVIEW`.
- Destructive actions are authorized at apply time, not by the saved plan alone.
- The UI stores a running operation status in session state and blocks further
  actions while a scan or apply operation is active.

## Verification

Use the focused regression tests:

```powershell
python -m pytest tests/test_file_resync.py
```

Useful manual checks:

- Run Scan & Plan with a narrow root and confirm content with aliases outside
  that root is not marked orphaned.
- Move a canonical file to one new path and confirm `SET_CANONICAL` is `SAFE`.
- Copy the same file to two paths and confirm canonical selection is `REVIEW`.
- Enable destructive apply only after exporting/reviewing the plan.

## Known Limitations

- Scans checksum every eligible file, which can be expensive on large roots.
  A future improvement should use size/mtime or inventory metadata before
  hashing unchanged files.
- Cancel currently clears the UI operation status. It does not interrupt a
  synchronous scan or apply call mid-execution. True cancellation requires a
  background task with cancellation checkpoints.
