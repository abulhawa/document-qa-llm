# Evaluation guide

The repository separates software tests from retrieval and QA quality evaluation. Existing JSON/CSV files under `docs/runbooks/` are historical experiment artifacts, not general product benchmarks.

## Retrieval evaluation

`scripts/run_retrieval_eval.py` evaluates the checked-in query fixture and can report expected-document Hit@k and MRR-style ranking measurements. It requires populated backends matching the fixture corpus. Review `--help` before running:

```bash
python scripts/run_retrieval_eval.py --help
```

Record the exact revision, fixture, service/model versions, flags, corpus state, output artifact, and latency. Compare a candidate against the same baseline rather than comparing unrelated runs.

## QA handoff and answer support

`scripts/run_qa_handoff_eval.py` compares context-selection strategies and support labels without treating retrieval success as final-answer correctness:

```bash
python scripts/run_qa_handoff_eval.py --help
```

When changing retrieval, ranking, grounding, or prompting, report the relevant subset of Hit@1/3/5, MRR, nDCG (if available), answer support, citation accuracy, refusal/failure counts, and latency. A change should not be described as an improvement without a reproducible before/after result.

## CI and evaluation strategy

The project is heavily containerized for local deployment, but CI does not need to reproduce the full deployment stack for every change. Keep four concerns separate:

| Layer | Purpose | Typical trigger | Target wall-clock time |
|---|---|---|---:|
| Unit/UI CI | Verify application logic and deterministic behavior | Every PR and push | under 5 minutes |
| Container integration smoke | Verify Docker/service wiring and ingestion plumbing | Relevant infrastructure/backend changes | 5-10 minutes |
| Retrieval regression evaluation | Measure retrieval quality against a fixed benchmark snapshot | Relevant retrieval changes or manual run | 5-15 minutes |
| Full raw-document evaluation | Rebuild from source documents through parsing, chunking, embedding and indexing | Manual, release, or ingestion/model changes | 30+ minutes is acceptable |

The slowest layer should not block ordinary development unless the change actually affects that layer.

### Current GitHub Actions behavior

The current workflows are intentionally lighter than the full Docker Compose deployment:

- `.github/workflows/tests.yml` runs Python directly on the GitHub runner and excludes `tests/e2e/`.
- `.github/workflows/e2e.yml` is manually triggered.
- The current E2E job starts real OpenSearch and Qdrant containers.
- Streamlit, the stub embedding service, the stub LLM, Playwright, and the test process run directly on the GitHub runner.
- The E2E workflow seeds OpenSearch and Qdrant directly. It does not exercise Redis, the Celery worker container, real document ingestion, the real embedding model, the reranker, Phoenix, Flower, Docker Compose networking, or production-style volume mapping.
- The stub embedder validates application wiring, not semantic retrieval quality.

This means the existing E2E workflow is best understood as an application smoke test against real storage backends, not a full deployment or ML-quality test.

The E2E backend versions should eventually be aligned with the deployment versions in `docker-compose.yml`. That change is intentionally separate from the benchmark skeleton so the existing runtime is not changed before review.

## Mixed public benchmark

Use a fixed mixed benchmark instead of personal documents and instead of relying on only one domain.

The initial benchmark mix should draw selected, license-compatible material from established sources such as:

- Open RAGBench for scientific/technical PDF retrieval and hard negatives.
- OfficeQA for office, government, financial, table-heavy document questions.
- Selected BEIR datasets for additional retrieval domains when the source format fits the evaluation goal.
- Additional benchmarks may be added later only when they cover a real gap.

Do not merge these sources into an opaque dataset. Every document and query must retain provenance:

```text
source_benchmark
source_version
source_document_id
source_query_id
domain
question_type
modality
qrels
license/provenance metadata
```

Report per-source metrics first. A composite score may be reported as a convenience, but it must not hide weak performance on one benchmark behind an easier benchmark.

The benchmark source selection must be deterministic and versioned. Re-running the source stage with the same source revisions, filters, and selection seed must produce the same corpus manifest.


### Proposed composite-v1 composition

`composite-v1` is a project regression benchmark derived from established datasets. It is not presented as an official score on any source benchmark because several source corpora are deliberately subsetted to keep rebuild and CI costs practical.

| Track | Planned corpus | Planned evaluation queries | Purpose |
|---|---:|---:|---|
| Open RAGBench text-only | 200 PDFs: 80 positive PDFs + 120 published hard-negative PDFs | 160 | Scientific/technical PDFs, paraphrase/extractive retrieval, realistic negative documents |
| OfficeQA Full | All source PDFs needed by 50 selected questions plus distractor Treasury Bulletins, targeting about 150 PDFs | 50: 25 easy + 25 hard | Financial/government PDFs, tables, long documents, temporal similarity |
| BEIR NFCorpus | Full corpus, about 3.6K text documents | 100 fixed test queries | Biomedical/health retrieval with multiple relevant documents and graded-ranking pressure |
| MIRACL hard-negative subsets | German and Arabic positives plus up to 20 hard negatives per query | 100: 50 German + 50 Arabic | Multilingual retrieval using the project's multilingual embedding model |

Target total: **410 evaluation queries**, roughly **350 PDF documents plus a few thousand text/passages**, with exact counts recorded after deterministic source selection and de-duplication.

Selection rules:

- Open RAGBench: choose 80 positive PDFs across available arXiv categories, select two text-only questions per chosen positive document where possible, and add 120 of the benchmark's published hard-negative PDFs. Prefer a mix of extractive and abstractive questions where the source metadata allows it.
- OfficeQA: use OfficeQA Full rather than only Pro so the regression set contains both easy and hard cases. Select 25 easy and 25 hard questions across years/decades, include every referenced source PDF, then add unreferenced Treasury Bulletins as distractors to target about 150 PDFs. Do not include Pro V2 in v1 because its separate 1,435-document corpus and much larger PDF download materially increase the source-stage cost.
- NFCorpus: retain the full small corpus so retrieval difficulty is not altered by negative-document sampling. Select 100 fixed test queries for routine evaluation; source-native full-query evaluation remains possible.
- MIRACL: use German and Arabic because they exercise the multilingual model in two different scripts. Use the published hard-negative form rather than indexing the multi-million-passage full corpora. Select 50 fixed queries per language and retain positives plus up to the top 20 published hard negatives for each query.

The exact selected IDs must be materialized into the source artifact manifest. No workflow should randomly resample at runtime.

### Evaluation tiers

All tiers use the same corpus/index artifact. Only the number of evaluated queries changes, so the expensive corpus, chunk, embedding, and index stages are not duplicated.

| Tier | Query count | Intended use |
|---|---:|---|
| smoke | 40 | Fast validation: 10 Open RAGBench, 10 OfficeQA, 10 NFCorpus, 5 German MIRACL, 5 Arabic MIRACL |
| medium | 120 | Routine retrieval regression: 40 Open RAGBench, 20 OfficeQA, 30 NFCorpus, 15 German MIRACL, 15 Arabic MIRACL |
| full | 410 | Manual/release evaluation using all composite-v1 queries |

Smoke and medium IDs must be fixed subsets of the full set, selected once and stored in the source manifest. They must never be sampled dynamically during CI.

Report metrics separately for Open RAGBench, OfficeQA, NFCorpus, MIRACL German, and MIRACL Arabic. If a single composite number is shown, use a macro-average across tracks so a source with more questions cannot dominate the result.

### Source-native benchmark runs

Where practical, keep adapters capable of running a source benchmark with its native corpus/query set. These runs are separate from `composite-v1` and are the appropriate place for comparisons against published benchmark results.

The composite benchmark optimizes for project regression coverage and CI practicality. Source-native runs optimize for external benchmark comparability.


### Deterministic source-selection policy

Benchmark membership must never depend on Document QA retrieval scores, embedding similarity, reranker output, or a previous evaluation result. Selection uses upstream labels/metadata plus a fixed SHA-256 ordering (`stable-sha256-v1`) seeded by the benchmark specification.

This avoids two failure modes:

- accidentally choosing examples that the current system already retrieves well
- changing benchmark membership when retrieval code changes

Open RAGBench selection:

1. Read `queries.json`, `qrels.json`, and the document IDs from `pdf_urls.json`.
2. Keep only queries whose upstream `source` is `text`.
3. Group those queries by their gold `doc_id`.
4. Keep positive documents with at least two eligible text-only queries.
5. Select 80 positive document IDs by stable SHA-256 ordering.
6. For each selected positive document, prefer one extractive and one abstractive query when both types exist; otherwise choose two eligible queries by the same stable ordering.
7. Derive the hard-negative pool as document IDs that are not a gold document for any query, then select 120 by stable ordering.
8. Do not select documents based on Document QA retrieval performance. The resulting arXiv category distribution is recorded in the manifest for audit rather than optimized after seeing scores.

The upstream benchmark explicitly distinguishes 400 positive documents, 600 hard-negative documents, and query generation source/type, so this policy uses those labels directly.

OfficeQA selection:

1. Load `officeqa_full.csv`.
2. Split questions by upstream `difficulty`.
3. Select 25 easy and 25 hard questions.
4. Within each difficulty class, round-robin across source-document decades; order candidates inside each decade with stable SHA-256.
5. Include every `source_file` required by the selected questions.
6. Fill the corpus to approximately 150 PDFs with unreferenced Treasury Bulletins.
7. Prefer distractors from the same decades represented by selected source documents, then fill any remaining capacity from the rest of the corpus.
8. Do not use answer values, retrieval scores, or model performance to choose distractors.

This keeps OfficeQA negatives temporally similar to the positives instead of making the task artificially easy through obvious date separation.

NFCorpus selection:

1. Keep the complete corpus.
2. Select 100 test-query IDs by stable SHA-256 ordering.
3. Retain the original qrels for those queries without altering relevance grades.

Keeping the full small corpus preserves the retrieval problem instead of weakening it by sampling negatives.

MIRACL German and Arabic selection:

1. Use the hard-negative repository's `test` split.
2. Select 50 query IDs per language by stable SHA-256 ordering.
3. Keep every published positive (`score > 0`) for each selected query.
4. Select up to 20 published hard negatives (`score <= 0`) per selected query by stable ordering.
5. The corpus for a language is the union of all selected positive and hard-negative candidate documents, so documents associated with other selected queries also act as additional negatives.

The composite MIRACL tracks are intentionally smaller project regression subsets, not official MIRACL scores.

The source manifest must record the exact selected IDs, source revisions, selection algorithm version, and seed. Once `composite-v1` is frozen, normal rebuilds consume those IDs directly rather than selecting them again.

## Successful source-build baseline

The first successful frozen source build was GitHub Actions run `36475735954` at repository revision `820d361df823cd9418ab78ad6b9da7a84f1aa2f2`.

| Track | Documents | Queries | Qrels |
|---|---:|---:|---:|
| Open RAGBench | 200 | 160 | 160 |
| OfficeQA | 150 | 50 | 89 |
| NFCorpus | 3,633 | 100 | 3,476 |
| MIRACL German | 430 | 50 | 517 |
| MIRACL Arabic | 456 | 50 | 502 |

The reconstructed source occupied 1,511,505,720 bytes on the runner. Only public Open RAGBench/NFCorpus download material is eligible for the Actions cache. The complete mixed corpus, including gated OfficeQA material, is persisted only in the private Hugging Face Storage Bucket.

## Evaluation artifact DAG

The full evaluation system is a dependency-aware artifact pipeline:

```text
1. source corpus
      ↓
2. parsed documents
      ↓
3. chunked documents
      ↓
4. embeddings
      ↓
5. OpenSearch + Qdrant indexes
      ↓
6. evaluation results
```

Every stage is reproducible, but ordinary evaluation should start from the newest compatible downstream artifact instead of recomputing upstream work.

### Stage 1: source corpus

The source artifact contains the fixed mixed benchmark inputs needed to reproduce the evaluation corpus:

```text
benchmark-source/
├── manifest.json
├── queries/
├── qrels/
├── provenance/
└── documents/ or source references permitted by each benchmark
```

Raw PDFs should be cached where licensing/access terms permit. For gated or restricted sources, store them privately or retain reproducible source references/checksums rather than republishing them.

### Stage 2: parsed documents

Rebuild when the source corpus, parser, loader, or parser-affecting preprocessing changes.

Current parse-stage layout:

```text
parsed/
├── <source-track>/
│   ├── documents.jsonl
│   ├── pages.jsonl
│   └── evaluation/
│       ├── queries.jsonl
│       ├── qrels.jsonl
│       └── answers.jsonl   # where supplied upstream
├── errors.jsonl
└── manifest.json
```

The document rows contain file-level parse statistics and source checksums; page rows carry cleaned page text plus normalized loader metadata. Runner-local absolute paths are replaced with stable `benchmark://...` logical sources.

### Stage 3: chunks

Rebuild when chunking or chunk-affecting preprocessing changes.

Typical contents:

```text
chunks/
├── chunks.jsonl
├── chunk_metadata.jsonl
└── manifest.json
```

For the normal case, parsing and chunking should be reused for a long time rather than repeated during every retrieval experiment.

### Stage 4: embeddings

Rebuild when chunks, the embedding model/revision, dimensionality, normalization, or embedding configuration changes.

The current embedding stage stores normalized multilingual E5 vectors in deterministic CPU/float32 shards and also embeds benchmark queries with the matching query role:

```text
embeddings/<fingerprint>/
├── shards/<NNN>/
│   ├── embeddings.npy
│   ├── records.jsonl
│   └── manifest.json
├── queries/<track>/
│   ├── embeddings.npy
│   └── records.jsonl
└── manifest.json
```

Document chunks use the E5 `passage: ` role and benchmark queries use `query: `. The exact model revision, runtime packages, normalization, input contract, execution device, and output dtype are included in artifact lineage.

Benchmark 4 is resumable at two levels. A complete shard is reused whenever its final shard manifest matches the embedding artifact fingerprint and shard layout, so execution-only workflow changes do not force already completed embeddings to run again. Incomplete shards are split into durable document checkpoints, currently 2,048 rows by default. Each checkpoint uploads its data first and its manifest last under `embeddings/<fingerprint>/work/<checkpoint-signature>/...`; the manifest is the completion marker. A checkpoint is reused only when both the embedding fingerprint and the checkpoint execution signature match. The execution signature tracks the checkpoint helper implementation, shard count, checkpoint size, and batch size. After all parts are present they are assembled into the unchanged final `shards/<NNN>/` contract, so downstream index/evaluation stages do not depend on checkpoint internals.

The chunk artifact plus embeddings form the portable canonical representation used to rebuild search-engine-specific indexes.

### Stage 5: native search indexes

Benchmark 5 builds native OpenSearch and Qdrant snapshots from the portable chunk and embedding artifacts. The two engines are built and persisted independently so one successful engine is never rebuilt merely because the other engine failed.

The engine identities deliberately have different invalidation boundaries:

- OpenSearch depends on the chunks artifact, pinned OpenSearch version, and the production chunk-index mapping/settings contract. It does not depend on the embedding artifact.
- Qdrant depends on both chunks and embeddings, the pinned Qdrant version, vector dimension/distance, and the minimal production payload contract (`id`, `checksum`, `path`).

This means an embedding-model change can reuse an unchanged OpenSearch snapshot while rebuilding only Qdrant.

Completed engine artifacts are immutable and reused by engine fingerprint. Initial builds also use coarse resumable checkpoints, 50,000 indexed rows by default. A checkpoint is a real engine-native snapshot: an OpenSearch filesystem-repository snapshot or a Qdrant collection snapshot. Checkpoint data and manifest are uploaded before `work/current.json` is advanced, so a failed new checkpoint does not replace the previous good recovery point.

Checkpoint compatibility is intentionally separate from final engine identity. Partial checkpoints require both the semantic engine fingerprint and a checkpoint signature derived from the current index-builder implementation. Therefore a change to checkpoint/recovery mechanics invalidates partial work without invalidating an already completed native index whose actual data/schema contract is unchanged.

Persisted layout is conceptually:

```text
indexes/
├── opensearch/<engine-fingerprint>/
│   ├── snapshot.tar.gz
│   ├── manifest.json
│   └── work/
│       ├── current.json
│       └── checkpoints/<rows>/...
├── qdrant/<engine-fingerprint>/
│   ├── snapshot.snapshot
│   ├── manifest.json
│   └── work/
│       ├── current.json
│       └── checkpoints/<rows>/...
├── artifacts/<combined-index-fingerprint>/manifest.json
└── current.json
```

The combined index manifest records the exact chunk and embedding parents plus the independently addressable OpenSearch and Qdrant engine artifacts.

Benchmark 5 pins:

```text
OpenSearch 3.8.0
Qdrant 1.19.1
```

Native backend snapshots are version-coupled performance artifacts. Benchmark 6 must restore them with compatible pinned backend versions. The chunk and embedding artifacts remain the canonical portable representation; if a native snapshot becomes incompatible after a backend upgrade, rebuild only the affected index stage rather than reparsing or re-embedding the corpus.

### Stage 6: evaluation

The common retrieval evaluation path should restore compatible native indexes, run the application retrieval pipeline, and calculate metrics.

Normal retrieval evaluation should not require:

- PDF parsing
- chunking
- document embedding
- Celery
- Redis
- Streamlit
- Flower
- Phoenix
- an answer-generation LLM

Use the real retrieval code and real document embeddings. Generation evaluation remains a separate layer.

## Artifact lineage and fingerprints

Every stage must include a manifest that identifies its direct parent and all configuration that can affect its output.

At minimum record:

- artifact type and schema version
- parent artifact ID/fingerprint
- benchmark source names, revisions, filters, and subset
- source-document checksums where available
- parser implementation/configuration fingerprint
- preprocessing fingerprint
- chunking strategy, size, and overlap
- embedding model name and revision
- embedding dimension and normalization
- OpenSearch/Qdrant versions and index schema fingerprint for index artifacts
- repository revision used to build the artifact
- creation timestamp for traceability, but not as the compatibility key

Compatibility must be determined from functional inputs, not from a manually chosen artifact name or timestamp.

If a required parent fingerprint does not match, the downstream workflow must fail clearly rather than silently use stale data.

## Invalidation and rebuild rules

Use the narrowest rebuild that preserves correctness:

| Change | First stage to rerun |
|---|---|
| Retrieval fusion, MMR, filters, top-k, ranking policy | evaluation |
| Query rewriting/planning | evaluation |
| Reranker | evaluation, unless its model artifact is separately cached |
| OpenSearch/Qdrant mapping/index configuration | index |
| OpenSearch/Qdrant version | index |
| Embedding model/configuration | embed |
| Chunker/chunk-affecting preprocessing | chunk |
| Parser/loader | parse |
| Benchmark source selection/revision | source |

After rerunning a stage, all downstream artifacts are considered stale until deliberately rebuilt.

Do not automatically cascade expensive downstream rebuilds while the implementation is still being reviewed. Manual stage boundaries keep Actions usage predictable and avoid rebuilding several expensive layers while an upstream change is still experimental.

## Manual benchmark workflows

The repository contains a manual workflow for each stage:

```text
.github/workflows/benchmark-source.yml
.github/workflows/benchmark-parse.yml
.github/workflows/benchmark-chunk.yml
.github/workflows/benchmark-embed.yml
.github/workflows/benchmark-index.yml
.github/workflows/benchmark-eval.yml
```

The **source** workflow is implemented but remains manual-only. It resolves Hugging Face repositories to immutable commit SHAs, applies the deterministic selection policy, downloads only the selected PDFs/text records, and writes a source lock and manifest. OfficeQA/Open RAGBench use `huggingface-hub`; NFCorpus is read from the official BEIR ZIP; MIRACL qrels/queries/corpus are read directly from pinned Parquet shards with PyArrow. The full Hugging Face `datasets` package is intentionally not installed. GitHub Actions obtains a short-lived read-only Hugging Face token through the account CI/CD OIDC identity, then persists the complete mixed source artifact privately under `hf://buckets/abulhawa/document-qa-artifacts/<benchmark>/source/<fingerprint>/`. A small mutable `source/current.json` pointer identifies the current immutable source artifact.

The parse workflow is operational and manual-only. It restores the current frozen source artifact from the private bucket, validates that its source lock and composition match the committed benchmark definition, runs the application's real `PyPDFLoader`/`TextLoader` path plus the existing document preprocessing, and writes deterministic per-document and per-page JSONL with a lineage manifest. For the frozen, trusted benchmark corpus only, the parser raises pypdf's Form XObject traversal cap from the production default of 5,000 to a bounded 50,000; this benchmark-only parser setting is recorded in the manifest and artifact fingerprint, while normal application ingestion keeps pypdf's default protection. Successful parsed artifacts are persisted privately under `hf://buckets/abulhawa/document-qa-artifacts/<benchmark>/parsed/<fingerprint>/`. Benchmark 3 is also operational and manual-only: it restores an exact parsed fingerprint, runs the production `core.chunking.split_documents()` implementation with explicit chunk size/overlap, records quality counts, and persists a content/configuration-addressed chunk artifact under `.../<benchmark>/chunks/<fingerprint>/` plus a mutable `chunks/current.json` pointer. GitHub Actions uses short-lived OIDC credentials: the Source workflow uses the account CI/CD identity for gated upstream reads and the bucket Trusted Publisher for persistence; downstream stages use the bucket identity to restore and publish artifacts. GitHub workflow artifacts still contain only small manifests and diagnostics. Benchmark 4 embedding and Benchmark 5 native indexing are operational and manual-only; the evaluation workflow remains a non-operational planning shell. No benchmark workflow is triggered automatically.


### Source-workflow dependency footprint

Keep acquisition dependencies narrower than ML/runtime dependencies. The source workflow installs only:

- `huggingface-hub` for immutable Hugging Face revisions and file downloads
- `pyarrow` for MIRACL Parquet shards
- `requests` for public PDF/BEIR downloads
- `PyYAML` for the benchmark composition file

Do not install `datasets`, pandas, embedding libraries, or application dependencies in the source workflow. They belong to later stages if needed.

`composite-v1` is frozen from successful source build run `36475735954`. The canonical machine lock is `evaluation/benchmarks/composite_v1.lock.json.gz`; `composite_v1.lock.json` is a small human-readable pointer/summary. The frozen lock records exact upstream revisions and selected IDs. Source reconstruction therefore no longer resolves or resamples benchmark membership. Raw mixed source is reconstructed by Benchmark 1 and persisted privately in the Hugging Face Storage Bucket. Benchmark 2 and later stages restore their exact parent artifact instead of rebuilding upstream stages.

## Artifact storage

Do not use ordinary GitHub Actions cache as the only durable source of benchmark artifacts. Actions cache is useful for disposable acceleration but can be evicted.

The intended storage model is:

```text
committed source lock + upstream immutable revisions   <- canonical source definition
public-only download cache                              <- optional acceleration
private durable artifact store                          <- source/parsed/chunks/embeddings/index
```

The private Hugging Face Storage Bucket `abulhawa/document-qa-artifacts` is the durable store for every canonical benchmark stage, including the reconstructed source corpus. Paths are content/configuration fingerprinted rather than timestamped, so downstream workflows can restore an exact compatible parent. GitHub Actions artifacts remain useful for small manifests and diagnostics but are not the system of record for corpus-bearing outputs.

For `composite-v1`, keep corpus-bearing artifacts private by default. OfficeQA is gated, and other benchmark sources have their own redistribution terms. **Do not put OfficeQA PDFs, CSV answer keys, or the mixed source bundle in GitHub Actions cache for this public repository.** Actions cache is not an appropriate privacy boundary for gated benchmark content. Only clearly public/reconstructible downloads may use Actions cache. The public repository should contain the composition specification, source/revision metadata, checksums/fingerprints where appropriate, workflow code, and evaluation results, but not assume that every upstream document or answer key can be republished.

The durable layout is:

```text
hf://buckets/abulhawa/document-qa-artifacts/
└── composite-v1/
    ├── source/
    │   ├── current.json
    │   └── <fingerprint>/
    ├── parsed/<fingerprint>/
    ├── chunks/<fingerprint>/
    ├── embeddings/<fingerprint>/
    ├── qdrant/<fingerprint>/
    └── opensearch/<fingerprint>/
```

The bucket is private. GitHub Actions uses Hugging Face Trusted Publisher / account CI/CD OIDC identities restricted to `abulhawa/document-qa-llm` on `refs/heads/master`. No long-lived Hugging Face write token is stored in GitHub. The source lock remains sufficient to reconstruct raw benchmark inputs if the bucket is lost.

## Trigger policy

Avoid running evaluation because an unrelated file changed.

Examples:

- Documentation or cosmetic UI changes: unit/UI checks only.
- Retrieval-policy changes: unit tests plus the small cached retrieval benchmark when enabled.
- Index mapping or backend-version changes: rebuild index artifact, then evaluate.
- Embedding changes: rebuild embeddings, index, then evaluate.
- Chunking changes: rebuild chunks, embeddings, index, then evaluate.
- Parser changes: rebuild parsed documents and every downstream stage.
- Benchmark composition changes: rebuild from source.
- Release candidate: run the larger/full evaluation appropriate to the release.

Independent jobs should run in parallel where possible so wall-clock CI time is determined by the slowest required layer rather than the sum of all layers.

## Container integration direction

A future container integration workflow should test the architecture that the current E2E job intentionally bypasses:

```text
small test document
      ↓
Redis/Celery
      ↓
real worker container
      ↓
real parser/chunker
      ↓
stub embedding service
      ↓
OpenSearch + Qdrant
      ↓
assert indexed state
```

The embedding and LLM services may remain deterministic stubs in this workflow because its purpose is deployment/integration correctness, not ML quality.

The retrieval benchmark should use real embeddings, but it does not need the full application stack when evaluating retrieval only.

## Safe fixtures and artifacts

Only commit synthetic, licensed, or otherwise redistributable fixtures. Remove personal paths, document contents, credentials, and identifiers. Store large/generated run outputs outside Git unless a small artifact is intentionally retained as a documented baseline.

Benchmark results should state:

- source benchmark and subset
- artifact fingerprints used
- backend/model versions
- whether native indexes were restored or rebuilt
- whether the run used the smoke, medium, or full evaluation set

This prevents a fast regression run from being mistaken for a complete end-to-end evaluation.
