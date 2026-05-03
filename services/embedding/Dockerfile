# Use an official Python runtime as a parent image
# We use a slim version to keep the image size small.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# We need build-essential for some python libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file first to leverage Docker cache
COPY pyproject.toml .

# Install dependencies
# We'll use 'pip install .' assuming pyproject.toml is set up
RUN pip install --no-cache-dir .

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
# We use uvicorn for a high-performance ASGI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
