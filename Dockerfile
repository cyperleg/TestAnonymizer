FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

WORKDIR /app

# System deps (kept minimal). Add build tools only if you really need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements*.txt ./

# Install Python deps
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

## Install PyTorch CPU wheel (GPU handled in a separate Dockerfile below)
#ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
#RUN pip install torch --index-url $TORCH_INDEX_URL

# Copy app
COPY . .

EXPOSE 8000

# Uvicorn in a single process to avoid duplicating model memory across workers.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
