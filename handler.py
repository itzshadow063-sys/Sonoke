"""
RunPod Serverless worker for Sonoke voice cloning.

How this fits in:
    Website  --(JSON: base64 voice sample + script + language)-->  RunPod API
        --> THIS FILE runs on a GPU worker that RunPod spins up automatically
        --> returns JSON: base64 of the generated WAV
        --> worker goes back to sleep (you stop paying)

Design notes:
    - Every startup step prints LOUDLY with flush=True, so if a worker
      crashes we can see exactly which line failed in the RunPod logs.
    - The heavy model load is wrapped in try/except that prints the full
      traceback before re-raising. (Previously a silent top-level crash gave
      us "exit code 1" with no explanation.)
    - The model loads LAZILY on the first request, not at import time. This
      means the container starts healthy immediately, and any model problem
      shows up as a clean job error instead of an unhealthy worker.
"""

import base64
import os
import re
import subprocess
import sys
import tempfile
import traceback


def log(msg):
    """Print with an immediate flush so RunPod captures it even on crash."""
    print(f"[sonoke] {msg}", flush=True)


log("Worker booting - importing base libraries...")

try:
    import numpy as np
    import runpod
    import soundfile as sf
    import torch
    log(f"Imports OK. numpy={np.__version__}, torch={torch.__version__}, "
        f"cuda_available={torch.cuda.is_available()}")
except Exception:
    log("FATAL: base import failed")
    traceback.print_exc()
    sys.stdout.flush()
    raise


# ----------------------------------------------------------------------
# Model is loaded lazily (on first job) so the worker starts healthy.
# ----------------------------------------------------------------------
MODEL_DIR = "/workspace/xtts_v2"
_model = None
_config = None


def _load_model():
    """Load XTTS-v2 once, cache it in module globals."""
    global _model, _config
    if _model is not None:
        return _model, _config

    log("First request - loading XTTS-v2 model...")

    # PyTorch 2.6+ defaults torch.load to weights_only=True which breaks
    # XTTS checkpoints. Force it off using the ORIGINAL loader (no recursion).
    try:
        _orig_load = torch.load

        def _safe_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = _safe_load

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            raise FileNotFoundError(
                f"{MODEL_DIR}/config.json not found - model was not baked "
                f"into the image. Contents: {os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else 'DIR MISSING'}"
            )

        cfg = XttsConfig()
        cfg.load_json(os.path.join(MODEL_DIR, "config.json"))
        mdl = Xtts.init_from_config(cfg)
        mdl.load_checkpoint(cfg, checkpoint_dir=MODEL_DIR, eval=True)
        mdl.cuda()

        _model, _config = mdl, cfg
        log("Model loaded. Worker ready to synthesize.")
        return _model, _config
    except Exception:
        log("FATAL: model load failed")
        traceback.print_exc()
        sys.stdout.flush()
        raise


# ----------------------------------------------------------------------
# The handler - runs once per incoming job
# ----------------------------------------------------------------------
def handler(event):
    """
    Expected event["input"]:
        {
          "voice_b64": "<base64-encoded audio file (mp3/wav)>",
          "voice_ext": "mp3",
          "script": "text to speak",
          "lang": "en"
        }
    Returns: { "audio_b64": "<base64-encoded WAV output>" }
    or on error: { "error": "<message>" }
    """
    try:
        job_input = event["input"]
        voice_b64 = job_input["voice_b64"]
        voice_ext = job_input.get("voice_ext", "mp3")
        script = job_input["script"]
        lang = job_input.get("lang", "en")

        model, config = _load_model()

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = os.path.join(tmp, f"sample.{voice_ext}")
            clean_path = os.path.join(tmp, "clean.wav")
            out_path = os.path.join(tmp, "output.wav")

            with open(raw_path, "wb") as f:
                f.write(base64.b64decode(voice_b64))

            # normalize the uploaded sample to what XTTS wants
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
                 "-ar", "22050", "-ac", "1", clean_path],
                check=True,
            )

            # sentence-chunk (XTTS caps ~400 tokens/call) then synthesize
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script.strip()) if s.strip()]
            if not sentences:
                sentences = [script.strip()]

            log(f"Synthesizing {len(sentences)} sentence(s), lang={lang}")
            sr = 24000
            gap = np.zeros(int(0.3 * sr), dtype=np.float32)
            pieces = []
            for i, sentence in enumerate(sentences):
                result = model.synthesize(
                    sentence, config, speaker_wav=clean_path, language=lang
                )
                pieces.append(np.asarray(result["wav"], dtype=np.float32))
                pieces.append(gap)

            full_audio = np.concatenate(pieces)
            sf.write(out_path, full_audio, sr)

            with open(out_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        log("Job complete.")
        return {"audio_b64": audio_b64}

    except Exception as e:
        log("Job failed")
        traceback.print_exc()
        sys.stdout.flush()
        return {"error": str(e)}


log("Starting RunPod serverless loop...")
runpod.serverless.start({"handler": handler})
