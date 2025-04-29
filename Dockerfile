# Use the NVIDIA CUDA Debian runtime image
FROM nvidia/cuda:12.2.0-runtime

# Install system dependencies (including ninja-build for flash-attn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg ca-certificates ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY requirements.txt .

# Install Python deps (ensure PyTorch CUDA version matches image)
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Entrypoint: launch Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]