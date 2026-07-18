# Base image: NVIDIA's official CUDA devel image (this tag definitely exists).
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# System dependencies: Python + ffmpeg (ffmpeg normalizes uploaded voice
# samples; Ubuntu 22.04 ships Python 3.10 by default)
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch + torchaudio, matched versions, for CUDA 12.1.
# - torchaudio is REQUIRED (coqui-tts/XTTS import it).
# - torch 2.5.1 (NOT 2.4.0): the modern transformers that coqui-tts pulls
#   imports `DTensor` from torch.distributed.tensor, which only exists in
#   torch >= 2.5. On torch 2.4 every job crashed with
#   "cannot import name 'DTensor'". 2.5.1 provides it.
RUN pip3 install --no-cache-dir \
    torch==2.5.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Fail the build EARLY (before the slow 2GB model download) if torchaudio
# didn't land - gives a clear, fast signal instead of a runtime crash.
RUN python3 -c "import torch, torchaudio; print('TORCH', torch.__version__, 'TORCHAUDIO', torchaudio.__version__)"

# Pin numpy < 2 (torch/coqui-tts are built against numpy 1.x).
RUN pip3 install --no-cache-dir "numpy<2"

# Voice cloning + serverless deps. We let coqui-tts pull its intended
# transformers (5.x); torch 2.5.1 above provides the DTensor it needs.
RUN pip3 install --no-cache-dir \
    runpod \
    coqui-tts \
    soundfile

# Re-assert numpy < 2 in case a dependency above bumped it back up.
RUN pip3 install --no-cache-dir "numpy<2"

# Bake the XTTS-v2 model into the image so cold starts don't re-download 2GB.
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('coqui/XTTS-v2', local_dir='/workspace/xtts_v2')"

# Confirm the model files landed.
RUN test -f /workspace/xtts_v2/config.json && echo "MODEL OK" || (echo "MODEL MISSING" && exit 1)

# Your worker code
COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
