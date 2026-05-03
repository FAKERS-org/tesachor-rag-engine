# Orchestration API Dockerfile

# Orchestrator (FastAPI) Service Dockerfile
FROM python:3.11-slim

# Copy the official uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Python best practices & uv configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install system dependencies
# libpq-dev is required for the psycopg2 driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libpq-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application code
# Note: In our structure, 'app/' is a subdirectory here
COPY . .

# Create a non-root user for the API
RUN groupadd -r apiuser && \
    useradd -r -m -g apiuser apiuser && \
    chown -R apiuser:apiuser /app

# Drop root privileges
USER apiuser

# Expose the API port
EXPOSE 8000

# Run the FastAPI app with uvicorn
# We use --host 0.0.0.0 to make it accessible outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
