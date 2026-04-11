FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY models.py .
COPY server/ server/
COPY tasks/ tasks/
COPY data/ data/
COPY datasets/ datasets/
COPY ui/ ui/
COPY tools/ tools/
COPY openenv.yaml .
COPY __init__.py .

# Copy pre-computed benchmark results if they exist
COPY outputs/ outputs/

# Create output directories
RUN mkdir -p outputs/sandbox outputs/logs outputs/evals outputs/benchmark outputs/episodes

EXPOSE 7860

# Default: run the environment server
# Override with: CMD ["python", "-m", "ui.app", "--port", "7860"]
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--ws-ping-interval", "60", "--ws-ping-timeout", "120"]
