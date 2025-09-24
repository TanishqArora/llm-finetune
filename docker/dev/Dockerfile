# CUDA + PyTorch runtime for training & notebooks
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Basic tools
RUN apt-get update && apt-get install -y git curl nano build-essential && rm -rf /var/lib/apt/lists/*

# Python libs for finetuning and notebooks
RUN pip install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    datasets \
    transformers \
    peft \
    accelerate \
    bitsandbytes \
    sentencepiece \
    trl \
    unsloth \
    scikit-learn \
    matplotlib \
    pandas \
    numpy \
    requests

# Optional: Hugging Face CLI (login in notebook or pass HF_TOKEN env)
RUN pip install --no-cache-dir huggingface_hub

WORKDIR /workspace