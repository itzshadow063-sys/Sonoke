# Base image: NVIDIA's official CUDA devel image - this is a very
# standard, well-known image that definitely exists (unlike the
# guessed RunPod-specific tag that failed before).
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# System dependencies: Python + ffmpeg (ffmpeg normalizes uploaded voice
# samples; Ubuntu 22.04 ships Python 3.10 by default)
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch, matched to this image's CUDA 12.1 - same install method that
# already worked reliably on your RunPod pod earlier.
RUN pip3 install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Voice cloning + serverless dependencies
RUN pip3 install --no-cache-dir \
    runpod \
    coqui-tts \
    "transformers>=4.57,<5.0" \
    soundfile \
    numpy

# Bake the XTTS-v2 model into the image at build time so cold starts
# don't need to download ~2GB from HuggingFace on every new worker.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('coqui/XTTS-v2', local_dir='/workspace/xtts_v2')"

# Your worker code
COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
