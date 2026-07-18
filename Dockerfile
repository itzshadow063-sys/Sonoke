# Base image: NVIDIA's official CUDA devel image (this tag definitely exists).
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# System dependencies: Python + ffmpeg (ffmpeg normalizes uploaded voice
# samples; Ubuntu 22.04 ships Python 3.10 by default)
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch matched to CUDA 12.1.
RUN pip3 install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# CRITICAL: pin numpy < 2. torch 2.4 and coqui-tts are compiled against
# numpy 1.x; numpy 2.x makes the workers crash at import time with
# "module compiled using NumPy 1.x cannot be run in NumPy 2.x" -> exit code 1.
RUN pip3 install --no-cache-dir "numpy<2"

# Voice cloning + serverless dependencies.
# NOTE: we do NOT force a transformers version anymore - we let coqui-tts
# pull the transformers release it was actually tested against. Forcing a
# very new transformers broke XTTS's internal GPT inference.
RUN pip3 install --no-cache-dir \
    runpod \
    coqui-tts \
    soundfile

# Re-assert numpy < 2 in case a dependency above bumped it back to 2.x.
RUN pip3 install --no-cache-dir "numpy<2"

# Bake the XTTS-v2 model into the image at build time so cold starts
# don't need to download ~2GB from HuggingFace on every new worker.
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('coqui/XTTS-v2', local_dir='/workspace/xtts_v2')"

# Quick sanity check at BUILD time: confirm the model files landed.
RUN test -f /workspace/xtts_v2/config.json && echo "MODEL OK" || (echo "MODEL MISSING" && exit 1)

# Your worker code
COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
