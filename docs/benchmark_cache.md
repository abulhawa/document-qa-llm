# Reusable benchmark cache indexes

`scripts/seed_benchmark_cache.py` derives reusable indexes from the frozen
`composite-v1` source, parsed, chunk, and embedding artifacts in the private
Hugging Face Storage Bucket. It does not rename, rewrite, or delete those
artifacts. Indexes are placed under `cache/` in the same bucket.

## Why the indexes use references

The v1 parser and chunker wrote track-wide JSONL files. The embedder wrote
112,724 vectors into eight NPY shards. Copying each vector to a separate bucket
object would make the cache expensive to enumerate and maintain. Instead, each
index row has a content-based key and a reference to an already persisted,
checksum-verified v1 artifact. Parsed and chunk entries point to a track file
plus a document ID; embedding entries point to a shard file and row number.
Consumers must verify the referenced file hash and select only the named
document or row. The indexes do **not** themselves change the existing v1
pipeline or make a v2 run incremental; cache-aware stage readers are needed next.
Some identical chunk texts have different stored vector bytes. The embedding
index retains every exact reference and reports how many text keys are
ambiguous. A future reader should prefer an exact stable chunk ID; it must
not silently choose one of several differing vectors for a text-only hit.

The parsed key includes the parser signature, file type, and source SHA-256.
File type matters because the loader can interpret identical bytes differently
under a different extension. The chunk key includes the chunker signature and
a digest of normalized parsed pages. The embedding key includes the model and
runtime signature plus the raw chunk-text SHA-256. Signatures include stage
code, configuration, package versions, and Python version, so a configuration
or implementation change cannot silently reuse incompatible output.

Each source file's declared hash and size are cross-checked against the source
manifest. The command downloads and hashes the consumed JSONL and NPY files
against their artifact manifests, checks document lineage and counts, validates
NPY dimensions, and joins each embedding record to its exact chunk. It does
not download every raw source document to remeasure its hash.

## Run and verify

Run the **Benchmark - Seed Reusable Cache** GitHub Actions workflow with
`apply=false` for a read-only validation. Set `apply=true` to upload the three
compressed index files and migration report. The workflow exchanges GitHub
OIDC for a bucket-scoped Hugging Face token; no long-lived token is committed.
The command also works from a trusted Python environment with
`huggingface-hub==2.0.0` and an HF token configured:

```bash
python -m scripts.seed_benchmark_cache
python -m scripts.seed_benchmark_cache --apply
```

The report is saved as `cache/migrations/<lineage-sha256>.json`. Its index
paths and SHA-256 values identify the exact files produced. Re-running the
same lineage verifies and reuses matching cache objects. It refuses to
overwrite an object if its bytes differ. A failed run can therefore be resumed
without deleting or changing frozen artifacts.

These indexes are a migration foundation. A future corpus version must resolve
cache keys, rewrite benchmark-specific paths and IDs, compute only cache misses,
and update indexes for additions and removals. Until those readers exist, the
current workflows still rebuild all stages for a changed corpus fingerprint.

