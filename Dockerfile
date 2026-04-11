FROM python:3.11-slim

WORKDIR /app

# System deps for lxml XML parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev libxslt-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files required by the environment server
COPY models.py .
COPY client.py .
COPY server/ server/
COPY datasets/ datasets/
COPY tools/ tools/
COPY openenv.yaml .
COPY __init__.py .

# Download clean datasets at build time (no CSVs in git)
RUN python tools/download_datasets.py

# Create output directories
RUN mkdir -p outputs/sandbox outputs/logs outputs/evals outputs/benchmark outputs/episodes

ENV SANDBOX_BASE=/app/outputs/sandbox
ENV DATA_DIR=/app/data

EXPOSE 7860

# Default: run the environment server
CMD ["python", "-c", "import uvicorn; import os; uvicorn.run('server.app:app', host='0.0.0.0', port=int(os.environ.get('PORT', '7860')), ws_ping_interval=60, ws_ping_timeout=120)"]
