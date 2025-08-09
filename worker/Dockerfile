FROM python:3.13-slim

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements/shared.txt requirements/worker.txt /tmp/req/

RUN pip install --no-cache-dir -r /tmp/req/worker.txt

COPY . .

CMD ["celery", "-A", "worker.celery_worker", "worker", "--loglevel=info", "--hostname=celery@%h", "--prefetch-multiplier=1"]