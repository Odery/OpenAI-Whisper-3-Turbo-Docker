# Use the NVIDIA CUDA Ubuntu base image
FROM nvidia/cuda:12.8.1-cudnn8-devel-ubuntu22.04

# Enable apt caching for faster rebuilds
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install system dependencies with cached apt
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    ca-certificates \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with cached pip
COPY requirements.txt .

# Install base packages first
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    "numpy<2" \
    python-multipart \
    packaging

# Install PyTorch with CUDA support
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    torch==2.2.2+cu121 \
    torchvision==0.17.2+cu121 \
    torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code (last step for optimal caching)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]