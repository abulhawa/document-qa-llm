# worker/Dockerfile
FROM python:3.13-slim

# --- OS/Base ---
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LANG=C.UTF-8

# System deps (kept minimal; add build tools only if you actually need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# --- Workdir ---
WORKDIR /app

# --- Dependency layer (cache-friendly) ---
# Keep the include relationship intact by copying both into the same dir
COPY requirements/shared.txt requirements/worker.txt /tmp/req/

# Upgrade pip toolchain first (helps with Py3.13 wheels)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir -r /tmp/req/worker.txt

# --- App code ---
COPY . .

# (Optional) run as non-root for security
# RUN useradd -m appuser && chown -R appuser:appuser /app
# USER appuser

# (Optional) healthcheck; replace module path if different
# HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
#   CMD celery -A worker.celery_worker inspect ping || exit 1

# Celery runtime
CMD ["celery", "-A", "worker.celery_worker", "worker", "--loglevel=info", "--hostname=celery@%h", "--prefetch-multiplier=1"]
