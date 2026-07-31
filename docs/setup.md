# Local setup and service operations

This guide documents the repository's current host-application plus Docker-service workflow. It does not promise a production deployment or a bundled LLM.

## Requirements

- Python 3.10 or newer. GitHub Actions currently runs Python 3.13.
- Docker Engine and Docker Compose v2 (`docker compose`, not the legacy `docker-compose` executable).
- Approximately 4 GB free RAM for OpenSearch and the host Python application, plus memory for Qdrant, Redis, the embedder, and the LLM. Corpus size and model choice can substantially increase RAM and disk requirements.
- The default `embedder-api` service requests an NVIDIA GPU and loads `intfloat/multilingual-e5-base`. Install a compatible NVIDIA driver and NVIDIA Container Toolkit. Initial startup downloads model weights and therefore needs network access and Hugging Face storage.
- A separately managed OpenAI-compatible LLM server. Model-specific RAM/VRAM requirements are not controlled by this repository; consult the selected model/server documentation.

## Python installation

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/app.txt -r requirements/worker.txt -r requirements/dev.txt
```

`app.txt` and `worker.txt` both include `shared.txt`. Installing all three commands is intentional for contributors who run the Streamlit process, worker modules, and tests from the same environment.

## Environment and documents

```bash
cp .env.example .env
mkdir -p sample_docs
```

Set `DOCUMENTS_PATH` in `.env` to the host directory that Celery may read. Compose mounts it read-only at `/documents`; the host Streamlit process sends paths which are translated through `DOC_PATH_MAP`. On Windows, use a path accepted by Docker Desktop and quote values containing spaces where your shell requires it.

Keep `.env` and document corpora out of Git. The checked-in `.env.example` contains placeholders and non-secret defaults only.

## Service startup

The service names defined by `docker-compose.yml` are:

```text
qdrant embedder-api phoenix opensearch opensearch-dashboards redis celery flower
```

Start the minimum services used for ingestion and retrieval:

```bash
docker compose up -d qdrant opensearch redis embedder-api celery
docker compose ps
```

Start local administration/observability UIs only when needed:

```bash
docker compose up -d phoenix opensearch-dashboards flower
```

Default host endpoints:

- Qdrant: `http://localhost:6333`
- embedding/reranking API: `http://localhost:8000`
- OpenSearch: `http://localhost:9200`
- Redis: `localhost:6379`
- Phoenix: `http://localhost:6006`
- OpenSearch Dashboards: `http://localhost:5601`
- Flower: `http://localhost:5555`

These development services are not authenticated. Bind/firewall them appropriately and do not expose them to an untrusted network.

## CPU-only embedding

The base Compose file is GPU-oriented. For a local CPU experiment, create an untracked `docker-compose.override.yml` that removes the GPU device reservation if your Compose implementation supports replacement and sets:

```yaml
services:
  embedder-api:
    environment:
      EMBEDDING_DEVICE: cpu
      EMBEDDING_FP16: "false"
```

Compose merging of device reservations varies by version; check the rendered result with `docker compose config`. If the reservation remains, maintain a local copy/override of the embedder service without the `deploy.resources.reservations.devices` block. CPU embedding is expected to be slower; no latency target is claimed.

## LLM startup

Run an OpenAI-compatible server separately and set `LLM_BASE_URL`. The default local client also uses text-generation-webui internal model list/load/info endpoints. If another server supports chat/completions but not those internal endpoints, explicitly set the `LLM_MODEL_*_ENDPOINT` variables or use a compatible adapter. Groq is optional and sends prompt/context data outside the local machine.

## Application startup

```bash
source .venv/bin/activate
streamlit run main.py
```

Use `docker compose logs --tail=100 <service>` when a dependency is unavailable. Stop containers without deleting data using `docker compose down`; add `--volumes` only when intentionally deleting local indexes, queues, model cache, and traces.

## Tests

```bash
python -m pytest --cov -q --ignore=tests/e2e --disable-warnings
python -m compileall -q app core ingestion qa_pipeline services ui utils worker
```

The E2E workflow is intentionally separate because it provisions real search/vector backends and Playwright. See `.github/workflows/e2e.yml` for the CI procedure.

## Verification status

Commands should be rechecked on every supported platform. The current public-readiness audit verifies Python dependency resolution and static/unit checks in the available environment. Docker commands cannot be considered locally verified when the Docker CLI/daemon is unavailable; `docker compose config` is the minimum validation to repeat on a Docker-capable host.
