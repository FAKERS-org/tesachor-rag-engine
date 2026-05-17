# LLM Service Dockerfile
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
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application code
COPY . .

# Create a non-root user for the API
RUN groupadd -r llmuser && \
    useradd -r -m -g llmuser llmuser && \
    chown -R llmuser:llmuser /app

# Drop root privileges
USER llmuser

# Expose the LLM service port
EXPOSE 8002

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
