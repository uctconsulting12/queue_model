# Use NVIDIA's PyTorch image with CUDA 12.1
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install OpenCV and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Expose port and run
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]


# docker run --gpus all -d   -p 8003:8000   -v "$USERPROFILE/.aws:/root/.aws"   --name queue_container   queue_image