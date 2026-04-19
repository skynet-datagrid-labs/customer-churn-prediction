FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/
COPY create_dummy_model.py .

# Create necessary directories
RUN mkdir -p artifacts/models artifacts/reports artifacts/metrics artifacts/plots

# Create dummy model
RUN python create_dummy_model.py

# Expose port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
