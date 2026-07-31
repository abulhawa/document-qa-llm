# Configuration reference

`config.py` loads `.env.local` (with override) and then `.env`; environment variables already present in the process take precedence over `.env`. Compose reads `.env` for `${...}` substitutions but explicitly sets container-internal service URLs where host URLs would be wrong.

The checked-in `.env.example` is the complete public template. No value in it is a production secret.

## Minimum interactive configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCUMENTS_PATH` | `./sample_docs` | Host folder mounted read-only into Celery |
| `WORKER_DOCUMENTS_PATH` | `/documents` | Corresponding container mount path |
| `DOC_PATH_MAP` | `./sample_docs=>/documents` | Host-to-worker path prefix mapping; set it explicitly with `DOCUMENTS_PATH` |
| `OPENSEARCH_URL` | `http://localhost:9200` | Host application OpenSearch endpoint |
| `QDRANT_URL` | `http://localhost:6333` | Host application Qdrant endpoint |
| `EMBEDDING_API_URL` | `http://localhost:8000/embed` | Embedding endpoint |
| `LLM_BASE_URL` | `http://localhost:5000` | Local LLM server base URL |
| `USE_GROQ` | `false` | Select the optional hosted Groq path |
| `GROQ_API_KEY` | empty | Required only when `USE_GROQ=true`; keep secret |

The LLM server is not included in Compose. The default model list/load/info paths are text-generation-webui-specific; override `LLM_MODEL_LIST_ENDPOINT`, `LLM_MODEL_LOAD_ENDPOINT`, and `LLM_MODEL_INFO_ENDPOINT` for another implementation.

## Storage isolation and indexes

`NAMESPACE`, `INDEX_PREFIX`, and `INDEX_SUFFIX` are combined with each `*_BASE` value. The bases cover chunks, full text, financial records, ingestion logs, inventory/watchlists, Qdrant chunks, and Qdrant file vectors. Changing `EMBEDDING_SIZE`, the embedding model, namespace, or collection/index bases can require reindexing; back up required data first.

## Embedding and chunking

- `EMBEDDING_MODEL_NAME`, `EMBEDDING_SIZE`, `EMBEDDING_BATCH_SIZE`, and `EMBEDDING_REQ_MAX_CHUNKS` configure the client/service contract.
- `EMBEDDING_DEVICE` and `EMBEDDING_FP16` configure the container model runtime.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `CHUNK_SCORE_THRESHOLD` affect indexing/retrieval behavior. Treat changes as experiments and evaluate/reindex consistently.

## Retrieval, QA, and caching

- Reranking is controlled by `RETRIEVAL_ENABLE_RERANK`, its candidate/top-N/timeouts, `RERANK_API_URL`, and the embedder service's `RERANK_*` model settings. It is off by default.
- Query planning and HyDE are controlled by `QA_ENABLE_QUERY_PLANNING` and `QA_ENABLE_HYDE`; both default off.
- `QA_HANDOFF_*` controls fixed or dynamic retrieval-to-prompt context packing.
- `QA_GROUNDING_ENABLED` defaults off; `QA_GROUNDING_THRESHOLD` is used when enabled.
- `LLM_CACHE_*` configures the OpenSearch response cache. With `LLM_CACHE_STORE_PROMPT_TEXT=true`, stored cache records may contain private document context.

These flags can change retrieval or answer behavior. Establish a baseline and use the evaluation guidance before describing a change as an improvement.

## Operations and tests

- `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, queues/service names, and the document path mapping control host/container task operations.
- `INGEST_*`, `OPENSEARCH_DELETE_BATCH`, `QDRANT_DELETE_BATCH`, and `OPENSEARCH_REQUEST_TIMEOUT` tune concurrency and backend operations.
- `CI`, `TEST_MODE`, `USE_STUB_EMBEDDER`, and `USE_STUB_LLM` isolate automated tests. Do not enable stubs for a real ingestion run.
- `PHOENIX_COLLECTOR_ENDPOINT` selects the host-process trace collector. The
  Compose worker uses the internal `http://phoenix:4317` endpoint. Phoenix is an
  optional Compose profile; traces may contain document/query metadata.

See `.env.example` for every supported public variable and its concrete default.
