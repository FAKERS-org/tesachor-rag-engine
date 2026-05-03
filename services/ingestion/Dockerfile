# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
# We need libpq-dev for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir .

# Copy the rest of the application code
COPY . .

# Environment variables will be provided by docker-compose
# The command is also usually overridden in docker-compose.yml 
# to start the Celery worker.
CMD ["celery", "-A", "worker", "worker", "--loglevel=info"]
