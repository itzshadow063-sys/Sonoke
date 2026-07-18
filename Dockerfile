# Base image: PyTorch 2.8 + CUDA 12.8, matches the version that worked
# cleanly on your RunPod pod (avoids the torch/torchaudio ABI hell you
# already fought through once).
FROM runpod/pytorch:2.8.0-py3.12-cuda12.8.1-devel-ubuntu22.04

WORKDIR /workspace

# System dependency: ffmpeg (used to normalize uploaded voice samples)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Python dependencies. transformers pinned to what coqui-tts needs -
# this container ONLY runs XTTS, so there's no LatentSync version
# conflict to worry about here.
RUN pip install --no-cache-dir \
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
