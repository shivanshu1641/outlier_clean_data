FROM python:3.11-slim

WORKDIR /app

# System deps: lxml, tini (zombie reaping init), sandboxuser (privilege separation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev libxslt-dev gcc tini \
    && useradd --no-create-home --shell /bin/false sandboxuser \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models.py .
COPY client.py .
COPY server/ server/
COPY ui/ ui/
COPY datasets/ datasets/
COPY tools/ tools/
COPY openenv.yaml .
COPY __init__.py .
COPY app.py .
COPY outputs/benchmark/ outputs/benchmark/

# Download clean datasets at build time (no CSVs in git)
RUN python tools/download_datasets.py

# Create output directories
RUN mkdir -p outputs/sandbox outputs/logs outputs/evals outputs/benchmark outputs/episodes

ENV SANDBOX_BASE=/app/outputs/sandbox
ENV DATA_DIR=/app/data
ENV ENABLE_WEB_INTERFACE=true

# Grant sandboxuser write access to sandbox output dir
RUN chown -R sandboxuser:sandboxuser /app/outputs/sandbox

EXPOSE 7860

# tini as PID 1: reaps zombie worker processes inside the container
ENTRYPOINT ["tini", "--"]
# Default: run the environment server with web interface
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--ws-ping-interval", "60", "--ws-ping-timeout", "120"]
