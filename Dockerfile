# Use the NVIDIA CUDA Ubuntu base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Configure APT and pip caching
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install system dependencies with cached apt
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-venv \
    python3-pip \
    ffmpeg \
    ca-certificates \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV --upgrade-deps
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV CUDA_HOME=/usr/local/cuda

# Upgrade build tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for cache optimization
COPY requirements.txt .

# (a) Install torch / torchaudio / torchvision first
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        python-multipart==0.0.7 \
        torch==2.2.2+cu121 \
        torchaudio==2.2.2+cu121 \
        torchvision==0.17.2+cu121 \
        --extra-index-url https://download.pytorch.org/whl/cu121

# (b) Then install everything else (including flash-attn)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-build-isolation \
        -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]