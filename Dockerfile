# Use the NVIDIA CUDA Debian runtime image
FROM nvidia/cuda:12.2.0-runtime-debian12

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY requirements.txt .

# Install Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Use the NVIDIA GPU at runtime
# (Docker 19.03+ supports --gpus; no extra runtime config needed)

# Entrypoint: launch Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
