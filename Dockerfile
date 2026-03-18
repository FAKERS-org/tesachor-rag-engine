# STAGE 1: The Build Environment
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

# 1. Enable Bytecode compilation for faster startup in 3.14
ENV UV_COMPILE_BYTECODE=1

# 2. Use 'copy' mode so we don't rely on hardlinks (which fail in some Docker drivers)
ENV UV_LINK_MODE=copy

WORKDIR /app

# 3. Cache the uv cache specifically. 
# We bind 'uv.lock' and 'pyproject.toml' so uv can resolve the environment 
# WITHOUT actually copying the files into the layer yet.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 4. Now we add the source code and perform the final sync
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# STAGE 2: The Runtime Environment (The "Production" Image)
FROM python:3.14-slim-bookworm

# 5. Copy ONLY the virtual environment from the builder
COPY --from=builder /app/.venv /app/.venv

# 6. Place the venv at the front of the PATH so 'python' and 'uvicorn' work globally
ENV PATH="/app/.venv/bin:$PATH"

# 7. Crucial for FastAPI logs to show up in Docker logs immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 8. Copy your source code (the 'src' directory)
COPY ./src /app/src

# 9. Execute using the module path. 
# Since we are in /app and our code is in src/main.py, the module is 'src.main'
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]