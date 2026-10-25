# Production Dockerfile for Wildfire Spread Prediction System
# Supports FastAPI REST microservice and Streamlit GIS Dashboard

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (OpenMP for LightGBM, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Layer-cached Python dependency installation
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and configurations
COPY configs/ configs/
COPY models/ models/
COPY src/ src/
COPY reports/ reports/

# Create non-root system user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Healthcheck for FastAPI microservice
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default execution: FastAPI REST microservice
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
