.PHONY: setup test check services app

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements/app.txt -r requirements/worker.txt -r requirements/dev.txt

test:
	python -m pytest --cov -q --ignore=tests/e2e --disable-warnings

check:
	python -m compileall -q app core ingestion qa_pipeline services ui utils worker

services:
	docker compose up -d qdrant opensearch redis embedder-api celery

app:
	streamlit run main.py
