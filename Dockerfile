# Use the Python 3.11 official image
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install project dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy local code to the container image
COPY . .

# Expose port (Railway sets PORT env variable)
EXPOSE 8080

# Run the web service on container startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

