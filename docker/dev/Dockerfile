# CUDA + PyTorch runtime for training & notebooks
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Avoid pulling in torchvision ops (like NMS) that we don’t need for text
ENV TRANSFORMERS_NO_TORCHVISION=1

# System deps
RUN apt-get update && apt-get install -y git curl nano build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements-dev.txt from project root into the image
COPY requirements-dev.txt /workspace/requirements-dev.txt

# Install everything
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements-dev.txt