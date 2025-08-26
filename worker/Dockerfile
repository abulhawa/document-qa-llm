FROM python:3.13-slim

# --- set local timezone inside the container ---
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*
 
# Set working directory
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements/shared.txt requirements/worker.txt /tmp/req/

RUN pip install --no-cache-dir -r /tmp/req/worker.txt

COPY . .

CMD ["celery", "-A", "worker.celery_worker", "worker", "--loglevel=info", "--hostname=celery@%h", "--prefetch-multiplier=1"]