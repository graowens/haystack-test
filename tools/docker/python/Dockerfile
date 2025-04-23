FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY tools/docker/python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["python", "app.py"]
