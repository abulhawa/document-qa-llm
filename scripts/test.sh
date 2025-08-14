#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Create a local venv to avoid global env bleed
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
# Ensure app environment installs cleanly
echo "Installing app requirements"
if ! python -m pip install -r requirements/app.txt; then
  echo "Failed to install app requirements" >&2
  exit 1
fi

# Ensure worker environment installs cleanly
echo "Installing worker requirements"
if ! python -m pip install -r requirements/worker.txt; then
  echo "Failed to install worker requirements" >&2
  exit 1
fi

# Install runtime deps used by tests + dev tools
python -m pip install -r requirements/shared.txt -r requirements/dev.txt

# Run tests via -m to avoid PATH issues
python -m pytest --cov -q --ignore=tests/e2e --disable-warnings
