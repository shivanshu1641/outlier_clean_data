FROM python:3.11-slim

WORKDIR /app

# System deps for lxml XML parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev libxslt-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[inference]"

# Copy project
COPY models.py .
COPY __init__.py .
COPY server/ server/
COPY datasets/ datasets/
COPY tools/ tools/
COPY client.py .
COPY openenv.yaml .

# Runtime dirs
RUN mkdir -p outputs/sandbox outputs/logs outputs/evals data

ENV SANDBOX_BASE=/app/outputs/sandbox
ENV DATA_DIR=/app/data

EXPOSE 7860

CMD ["python", "-c", "import uvicorn; import os; uvicorn.run('server.app:app', host='0.0.0.0', port=int(os.environ.get('PORT', '7860')))"]
