FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app
ENV PYTHONPATH=/app

# Install only what your FastAPI embedder app needs
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    sentence-transformers

# Copy application code
COPY app.py .
COPY config.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
