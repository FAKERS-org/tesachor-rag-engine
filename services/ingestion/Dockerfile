# Ingestion Service Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Copy the official uv binary (multi-stage copy is faster & more secure than curl)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Python best practices & uv configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/app/.venv/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application code
COPY . .

# Create a non-root user & group, then fix permissions
RUN groupadd -r celeryIngestionUser && \
    useradd -r -m -g celeryIngestionUser celeryIngestionUser && \
    chown -R celeryIngestionUser:celeryIngestionUser /app

# Drop root privileges for runtime
USER celeryIngestionUser

# Default command (usually overridden in docker-compose.yml)
CMD ["celery", "-A", "worker", "worker", "--loglevel=info"]