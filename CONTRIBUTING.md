# Contributing

Contributions are welcome. Bug fixes, tests, documentation, evaluation improvements, and focused feature proposals can all help this learning-oriented document-QA project.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). For vulnerabilities, use the private process in [SECURITY.md](SECURITY.md), not a public issue.

## Development setup

Prerequisites:

- Git and Python 3.10 or newer (CI currently exercises Python 3.13)
- Docker with the Compose plugin for integration services
- An NVIDIA GPU and NVIDIA Container Toolkit for the default embedding service; see [setup details](docs/setup.md) for CPU alternatives and realistic resource requirements
- An OpenAI-compatible local LLM server for interactive answers

```bash
git clone <repository-url>
cd document-qa-llm
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/app.txt -r requirements/worker.txt -r requirements/dev.txt
cp .env.example .env
```

The full service and application procedure is in [`docs/setup.md`](docs/setup.md). Never commit `.env`, local documents, model files, indexes, or credentials.

## Tests and checks

Run the unit/UI suite without external backends:

```bash
python -m pytest --cov -q --ignore=tests/e2e --disable-warnings
```

Run the repository's supported static syntax check:

```bash
python -m compileall -q app core ingestion qa_pipeline services ui utils worker
```

The E2E suite starts real OpenSearch and Qdrant services and uses stub model APIs; follow the CI workflow in `.github/workflows/e2e.yml` when changing integration behavior. This repository does not currently enforce an auto-formatter. Keep edits focused and follow the surrounding Python style rather than reformatting unrelated code.

## Branches and pull requests

1. Create a short-lived branch from the current default branch, such as `fix/clear-error-message` or `docs/setup-linux`.
2. Keep one concern per pull request; avoid drive-by refactors or generated formatting changes.
3. Explain the current behavior, the reason for the change, and any behavior or compatibility impact.
4. Add tests for changed behavior and run the relevant checks locally.
5. Update user or architecture documentation when interfaces, setup, configuration, or behavior change.
6. Complete the pull-request template and link related issues.
7. For retrieval, ranking, grounding, or answer-quality changes, include a baseline, the evaluation command/data, before/after metrics, latency where relevant, and remaining failure cases. Do not infer quality from a few examples.

Maintainers may ask for smaller commits, clearer evaluation evidence, or follow-up documentation before merging.

## Reporting bugs

Use the bug-report template and include reproducible steps, expected and actual behavior, relevant logs with secrets/documents removed, operating system, Python version, and service versions. Search existing issues first. Use private vulnerability reporting for security problems.

## Proposing features

Use the feature-request template. Describe the user problem before the implementation, expected scope, alternatives, and how success can be verified. Discuss large architecture or dependency additions before implementing them.

## Adding tests

- Put focused unit tests under `tests/` and name files/functions with the `test_` prefix.
- Mock network/model boundaries for unit tests; keep tests deterministic and independent of private documents.
- Add regression tests that fail before a bug fix and pass afterward.
- Mark backend-dependent tests with the existing `e2e` marker and do not add tests for Vercel or other cloud services.
- For retrieval changes, keep evaluation fixtures synthetic or redistributable and report retrieval and support metrics separately.
- Do not include credentials, personal paths, copyrighted document corpora, or production data in fixtures or snapshots.
