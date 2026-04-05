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
COPY openenv.yaml .
COPY __init__.py .

# Create outputs directory
RUN mkdir -p outputs/sandbox outputs/logs outputs/evals

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--ws-ping-interval", "60", "--ws-ping-timeout", "120"]
