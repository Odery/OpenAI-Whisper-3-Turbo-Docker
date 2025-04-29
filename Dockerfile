# Use the NVIDIA CUDA Debian runtime image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
       python3 python3-pip ffmpeg ca-certificates ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and pin Python packages
COPY requirements.txt .

# First, install NumPy<2 and python-multipart alongside packaging
RUN pip3 install --no-cache-dir \
       "numpy<2" \
       python-multipart \
       packaging

# Install PyTorch + friends (CUDA matched) and the rest
RUN pip3 install --no-cache-dir \
       torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
         --index-url https://download.pytorch.org/whl/cu121 \
         --extra-index-url https://pypi.org/simple \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
