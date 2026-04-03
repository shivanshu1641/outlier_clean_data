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

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
