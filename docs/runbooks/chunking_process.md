# Chunking Process

This document explains how raw extracted text becomes retrievable chunks during ingestion. It documents the current implementation only; it does not claim a retrieval-quality improvement.

## Current Data Flow

1. File loading creates LangChain `Document` objects with `page_content` and metadata such as `source`, `path`, and optionally `page`.
2. `ingestion.preprocess.preprocess_documents()` runs text preprocessing through `core.document_preprocessor.preprocess_to_documents()`. If preprocessing fails, ingestion logs a warning and keeps the original documents.
3. `ingestion.preprocess.build_full_text()` joins the document contents into one full-text payload for the full-text OpenSearch document.
4. `ingestion.preprocess.chunk_documents()` delegates chunk construction to `core.chunking.split_documents()`.
5. `core.chunking.split_documents()` uses LangChain's `RecursiveCharacterTextSplitter` with:
   - `CHUNK_SIZE` from `config.py` / environment, currently defaulting to `800`
   - `CHUNK_OVERLAP` from `config.py` / environment, currently defaulting to `100`
   - separators in this order: paragraph break, line break, space, then character fallback
6. `split_documents()` returns dictionaries with `text`, `page`, `path`, `chunk_index`, and `location_percent`.
7. `ingestion.orchestrator.ingest_one()` then applies file-level metadata to every chunk:
   - deterministic `id` from `uuid5(checksum:index)`
   - file-level `chunk_index`
   - canonical `path`
   - `checksum`
   - `chunk_char_len`
   - file type, timestamps, byte size, and formatted size
   - file-level `location_percent`
   - document identity metadata such as `doc_type` and `person_name`
   - financial metadata when extraction is available
8. `ingestion.storage.embed_and_store()` embeds the chunks, upserts dense vectors to Qdrant, and indexes chunk documents into OpenSearch through the batch callback. The orchestrator then indexes the full-text document and updates inventory metadata.

## Final Chunk Shape

The final indexed chunk is the orchestrator-enriched record, not the raw return value from `core.chunking.split_documents()`.

Important fields:

- `id`: deterministic chunk ID derived from file checksum and final chunk index.
- `text`: chunk text sent to the embedding service and stored for retrieval display.
- `chunk_index`: zero-based position of the chunk within the full ingested file.
- `location_percent`: approximate file-level position from `0` to `100`.
- `path`: canonical file path used for source tracking and replacement.
- `checksum`: content checksum used for deduplication and deterministic IDs.
- `page`: page metadata when the loader supplied it.
- `chunk_char_len`: character length of the final chunk text.

## Design Notes

- Chunking is character-based, not token-based or semantic.
- The recursive splitter tries to keep paragraph and line boundaries before falling back to smaller separators.
- Overlap intentionally duplicates nearby text so boundary-spanning answers can still retrieve enough context.
- Existing indexed files keep their previous chunk shape until re-ingested. Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` affects only new or re-ingested files.
- Mixed old and new chunk shapes can make retrieval behavior harder to interpret. Use `docs/runbooks/chunk_size_migration.md` for staged re-ingestion.

## Retrieval and Evaluation Implications

Chunking changes can affect:

- dense retrieval precision and recall
- BM25 matching in OpenSearch
- citation granularity
- prompt context packing
- duplicate evidence caused by overlap
- indexing and retrieval latency

Do not describe a chunking change as better unless an evaluation run supports it. For retrieval or answer-quality changes, compare before and after with the retrieval or QA handoff fixtures and report metrics such as Hit@1, Hit@3, Hit@5, MRR, nDCG, answer support, citation accuracy, and latency where available.

## How To Verify Changes

For documentation-only changes, code behavior does not need a retrieval evaluation.

For code or configuration changes to chunking, use the smallest relevant checks first:

```powershell
pytest tests/test_config_utils.py tests/test_ingestion_extra.py tests/test_ingest_fulltext.py -q
```

For retrieval-facing changes, run a baseline and after-change comparison:

```powershell
python scripts/run_retrieval_eval.py --fixture tests/fixtures/retrieval_eval_queries.json --support-labels tests/fixtures/retrieval_eval_answer_support_labels.json --output docs/runbooks/retrieval_eval_<label>.json
```

If chunk size or overlap changes, follow the staged operational process in `docs/runbooks/chunk_size_migration.md`.

## Known Limitations

- The splitter does not use embedding-aware or semantic boundary detection.
- The splitter does not account for LLM token limits directly.
- Page metadata depends on the upstream loader and may be absent.
- Empty or image-only documents can produce no valid chunks; OCR remains a separate ingestion concern.
- The current policy is global rather than per document type.
