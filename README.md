# Local Document Q&A

Local Document Q&A indexes documents and answers questions against retrieved evidence. Unlike a hosted chat-with-files service, its storage, retrieval, embedding, generation, and tracing endpoints are configurable and can run on infrastructure you control. The implemented pipeline combines OpenSearch lexical search with Qdrant dense retrieval, returns source metadata with answers, and exposes traces for inspecting pipeline behavior. Local deployment improves data control, but operators must still secure ports, logs, model services, and the host filesystem.

![Streamlit document Q&A interface](assets/screenshot_ui.png)

## Features

- PDF, DOCX, and text ingestion with chunk metadata and checksum/path deduplication
- Hybrid retrieval over OpenSearch and Qdrant, with configurable fusion, MMR, query variants, and an optional reranker
- OpenAI-compatible local LLM integration and an optional Groq provider
- Source filename/page or position metadata shown with generated answers
- Streamlit pages for chat, ingestion, search, index inspection, duplicates, watchlists, task administration, and topic discovery
- Celery/Redis asynchronous ingestion workers
- OpenTelemetry/Phoenix tracing for ingestion and QA pipeline inspection
- Retrieval and QA handoff evaluation scripts with checked-in evaluation fixtures and prior run artifacts

See [Architecture](docs/architecture.md) for component boundaries and [Pipeline map](docs/pipeline_map.md) for code-level flow.

## Architecture

```mermaid
flowchart LR
    D[Local documents] --> UI[Streamlit UI]
    UI --> C[Celery worker]
    C --> E[Embedding API]
    C --> OS[(OpenSearch)]
    C --> Q[(Qdrant)]
    UI --> R[Hybrid retrieval pipeline]
    OS --> R
    Q --> R
    R --> P[Prompt and grounding pipeline]
    P --> L[OpenAI-compatible LLM]
    L --> A[Answer and source citations]
    UI --> PH[Phoenix tracing]
    C --> PH
```

The repository does not bundle an LLM server. The default embedding container requires CUDA; CPU operation needs an override described in [Setup](docs/setup.md).

## Quick start

### Prerequisites

- Python 3.10+ (CI uses Python 3.13)
- Docker Engine with the `docker compose` plugin
- For the default embedder: NVIDIA GPU, compatible driver, and NVIDIA Container Toolkit
- An OpenAI-compatible LLM server reachable at `http://localhost:5000` by default
- At least 4 GB RAM for OpenSearch and the Python application in addition to model memory; actual model RAM/VRAM and disk use depend on the selected embedding and generation models

### 1. Install

```bash
git clone <repository-url>
cd document-qa-llm
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/app.txt -r requirements/worker.txt -r requirements/dev.txt
cp .env.example .env
mkdir -p sample_docs
```

### 2. Configure the minimum environment

For the default local endpoints, `.env.example` already contains usable non-secret values. Set `DOCUMENTS_PATH` to the host folder the worker may read and configure the LLM endpoint/model server. Do not commit `.env`.

```dotenv
DOCUMENTS_PATH=./sample_docs
LLM_BASE_URL=http://localhost:5000
USE_GROQ=false
```

All environment variables, defaults, and advanced switches are documented in [Configuration](docs/configuration.md).

### 3. Start services

Start the minimum data, queue, embedding, and worker services:

```bash
docker compose up -d qdrant opensearch redis embedder-api celery
```

Check their state and inspect failures before launching the UI:

```bash
docker compose ps
docker compose logs --tail=100 embedder-api celery
```

Phoenix tracing, OpenSearch Dashboards, and Flower are optional:

```bash
docker compose up -d phoenix opensearch-dashboards flower
```

### 4. Start the application

Start your separately managed OpenAI-compatible LLM server, then run:

```bash
streamlit run main.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`). For setup troubleshooting, CPU embedding, service ports, and shutdown commands, see [Setup](docs/setup.md).

> There is intentionally no one-command full startup claim: model selection, model download, GPU/CPU compatibility, and the external LLM server require operator choices that cannot be safely automated by this repository.

## Tests and checks

Install the requirements shown above, then run:

```bash
python -m pytest --cov -q --ignore=tests/e2e --disable-warnings
python -m compileall -q app core ingestion qa_pipeline services ui utils worker
```

E2E tests require Docker, OpenSearch, Qdrant, Playwright Chromium, and the repository's stub services. CI documents that orchestration in [`.github/workflows/e2e.yml`](.github/workflows/e2e.yml). Evaluation commands and the meaning of stored results are documented in [`docs/evaluation.md`](docs/evaluation.md).

## Minimum vs. optional services

| Component | Minimum interactive setup | Notes |
| --- | --- | --- |
| OpenSearch | Required | Lexical/full-text indexes; local Compose disables security |
| Qdrant | Required | Dense vectors and metadata |
| Redis + Celery | Required for asynchronous ingestion | Worker mounts only `DOCUMENTS_PATH` read-only |
| Embedding API | Required | Default Compose configuration uses CUDA |
| OpenAI-compatible LLM | Required for generated answers | External to this Compose file |
| Phoenix | Optional | Trace collection and inspection |
| OpenSearch Dashboards | Optional | OpenSearch inspection UI |
| Flower | Optional | Celery monitoring; unauthenticated in local Compose |
| Groq | Optional | Sends prompts/context to a third-party API; requires `GROQ_API_KEY` |
| Cross-encoder reranking | Optional, off by default | Served by the embedding API; adds model memory and latency |

## Project status

This is a pre-1.0 learning and portfolio project. It has unit/UI tests and evaluation tooling, but it is not presented as production-ready. Defaults are designed for a trusted local development machine: several service ports are host-accessible, OpenSearch authentication is disabled, Flower is unauthenticated, document-level authorization is not implemented, and generated answers still require source verification. Streaming answer display is currently disabled.

Past evaluation artifacts under `docs/runbooks/` describe specific fixtures and configurations; they are not general benchmarks or guarantees for other corpora.

The [public repository audit](docs/public-repository-audit.md) identifies personal path/filename metadata in historical evaluation artifacts that the owner must review before publication.

## Roadmap

- Make Compose profiles for CPU/GPU embedding and optional observability services
- Add authenticated/network-hardened deployment guidance
- Expand retrieval, answer-support, citation-accuracy, refusal, and latency evaluation coverage
- Define a stable pre-1.0 configuration and migration policy
- Improve accessibility and cross-platform setup verification

Roadmap items are proposals, not commitments. Discuss large changes in an issue before implementation.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests, pull-request expectations, and evaluation requirements. Community participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md); report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## License

Licensed under the [Apache License 2.0](LICENSE). Copyright 2026 Ali Abul Hawa.

## Acknowledgements

This project builds on open-source components including Streamlit, OpenSearch, Qdrant, Celery, Redis, Sentence Transformers, Arize Phoenix, LangChain document utilities, and the Python scientific ecosystem. Their names identify dependencies and do not imply endorsement.

AI coding tools have been used as implementation accelerators in parts of this learning project; architecture review, evaluation, debugging, and maintenance remain the project owner's responsibility.
