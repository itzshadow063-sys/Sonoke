"""
RunPod Serverless worker for Sonoke voice cloning.

How this fits in:
    Website  --(JSON: base64 voice sample + script + language)-->  RunPod API
        --> THIS FILE runs on a GPU worker that RunPod spins up automatically
        --> returns JSON: base64 of the generated WAV
        --> worker goes back to sleep (you stop paying)

The model is loaded ONCE per worker (at cold start), then reused for every
job that worker receives while it stays warm - this is why cold starts are
slower than the requests right after.
"""

import base64
import io
import os
import re
import subprocess
import tempfile

import numpy as np
import runpod
import soundfile as sf
import torch

# ----------------------------------------------------------------------
# 1. Load the XTTS model ONCE at worker startup (not per-request)
# ----------------------------------------------------------------------
print("Loading XTTS-v2 model...")


def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return torch.serialization.load(*args, **kwargs)


torch.load = _safe_load

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

MODEL_DIR = "/workspace/xtts_v2"  # baked into the Docker image (see Dockerfile)

config = XttsConfig()
config.load_json(os.path.join(MODEL_DIR, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, eval=True)
model.cuda()

print("Model loaded. Worker ready.")


# ----------------------------------------------------------------------
# 2. The handler - runs once per incoming job
# ----------------------------------------------------------------------
def handler(event):
    """
    Expected event["input"]:
        {
          "voice_b64": "<base64-encoded audio file (mp3/wav)>",
          "voice_ext": "mp3",          # file extension of the sample
          "script": "text to speak",
          "lang": "en"
        }

    Returns:
        { "audio_b64": "<base64-encoded WAV output>" }
    """
    job_input = event["input"]

    voice_b64 = job_input["voice_b64"]
    voice_ext = job_input.get("voice_ext", "mp3")
    script = job_input["script"]
    lang = job_input.get("lang", "en")

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, f"sample.{voice_ext}")
        clean_path = os.path.join(tmp, "clean.wav")
        out_path = os.path.join(tmp, "output.wav")

        # decode the uploaded voice sample
        with open(raw_path, "wb") as f:
            f.write(base64.b64decode(voice_b64))

        # normalize to the format XTTS wants
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
             "-ar", "22050", "-ac", "1", clean_path],
            check=True,
        )

        # sentence-chunk (XTTS caps at ~400 tokens per call) and synthesize
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script.strip()) if s.strip()]
        sr = 24000
        gap = np.zeros(int(0.3 * sr), dtype=np.float32)
        pieces = []
        for sentence in sentences:
            result = model.synthesize(sentence, config, speaker_wav=clean_path, language=lang)
            pieces.append(np.asarray(result["wav"], dtype=np.float32))
            pieces.append(gap)

        full_audio = np.concatenate(pieces)
        sf.write(out_path, full_audio, sr)

        with open(out_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"audio_b64": audio_b64}


# ----------------------------------------------------------------------
# 3. Start the RunPod serverless loop
# ----------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
