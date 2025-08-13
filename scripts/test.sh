#!/usr/bin/env bash
set -e

pip install -r requirements/shared.txt -r requirements/dev.txt
pytest --cov
