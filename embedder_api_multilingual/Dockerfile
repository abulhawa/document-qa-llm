# Base image with Python
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app
ENV PYTHONPATH=/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY config.py .


# Expose default FastAPI port (optional, helpful for clarity)
EXPOSE 8000

# Default command: start FastAPI app via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
