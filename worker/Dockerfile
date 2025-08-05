FROM python:3.13-slim

# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app

COPY worker/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["celery", "-A", "worker.celery_worker", "worker", "--loglevel=info", "--hostname=celery@%h"]