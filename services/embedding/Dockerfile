# Embedding Service Dockerfile

# official python
FROM python:3.11-slim

# copy official uv binary for faster startup
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# disable bytecode and unbuffer output for real time logging
# HF_HOME is for Hugging Face model caching;
# we don't to download models every time we build the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HOME=/app/model_cache \  
    PATH="/app/.venv/bin:$PATH"

# set working directory
WORKDIR /app

# install sys dependancies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# copy dependencies list
COPY pyproject.toml ./
COPY uv.lock ./

# install dependencies using uv
RUN uv sync --frozen --no-dev --no-install-project

# copy the rest of the code
COPY . .

# create non root user for security
RUN groupadd -r celeryEmbeddingUser && \
    useradd -r -m -g celeryEmbeddingUser celeryEmbeddingUser && \
    mkdir -p /app/model_cache && \
    chown -R celeryEmbeddingUser:celeryEmbeddingUser /app

USER celeryEmbeddingUser

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]