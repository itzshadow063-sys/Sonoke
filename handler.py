"""
RunPod Serverless worker for Sonoke voice cloning.
"""

import base64
import os
import re
import subprocess
import sys
import tempfile
import traceback


def log(msg):
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


MODEL_DIR = "/workspace/xtts_v2"
_model = None
_config = None


def _load_model():
    global _model, _config
    if _model is not None:
        return _model, _config
    log("First request - loading XTTS-v2 model...")
    try:
        _orig_load = torch.load

        def _safe_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)

        torch.load = _safe_load

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            raise FileNotFoundError(f"{MODEL_DIR}/config.json not found")

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


def chunk_text(text, max_len=220):
    text = (text or "").strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > max_len:
            if cur:
                chunks.append(cur.strip()); cur = ""
            piece = ""
            for w in s.split():
                if len(piece) + len(w) + 1 > max_len:
                    if piece:
                        chunks.append(piece.strip())
                    piece = w
                else:
                    piece = (piece + " " + w).strip()
            if piece:
                chunks.append(piece.strip())
        elif len(cur) + len(s) + 1 > max_len:
            chunks.append(cur.strip()); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        chunks.append(cur.strip())
    return [c for c in chunks if c]


def upload_audio(path):
    """Upload the MP3 to YOUR Supabase Storage (reliable) and return a
    public URL. Needs RunPod env vars SUPABASE_URL + SUPABASE_SERVICE_KEY
    and a PUBLIC bucket named 'audio'."""
    import requests
    import uuid

    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not base or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not set on RunPod")

    name = "gen-" + uuid.uuid4().hex + ".mp3"
    with open(path, "rb") as f:
        data = f.read()

    r = requests.post(
        f"{base}/storage/v1/object/audio/{name}",
        headers={
            "Authorization": f"Bearer {key}",
            "apikey": key,
            "Content-Type": "audio/mpeg",
            "x-upsert": "true",
        },
        data=data,
        timeout=300,
    )
    if not r.ok:
        raise RuntimeError(f"Supabase upload failed {r.status_code}: {r.text[:200]}")
    return f"{base}/storage/v1/object/public/audio/{name}"


def handler(event):
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

            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
                 "-ar", "22050", "-ac", "1", clean_path],
                check=True,
            )

            chunks = chunk_text(script, max_len=220)
            if not chunks:
                chunks = [script.strip()]

            log(f"Synthesizing {len(chunks)} chunk(s), lang={lang}")
            sr = 24000
            gap = np.zeros(int(0.3 * sr), dtype=np.float32)
            pieces = []
            for i, chunk in enumerate(chunks):
                result = model.synthesize(chunk, config, speaker_wav=clean_path, language=lang)
                pieces.append(np.asarray(result["wav"], dtype=np.float32))
                pieces.append(gap)
                log(f"  chunk {i+1}/{len(chunks)} done")

            full_audio = np.concatenate(pieces)
            sf.write(out_path, full_audio, sr)

            mp3_path = os.path.join(tmp, "output.mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", out_path,
                 "-b:a", "48k", mp3_path],
                check=True,
            )

            size = os.path.getsize(mp3_path)
            if size < 1_500_000:  # tiny clips return inline (fast); rest -> Supabase
                with open(mp3_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                out_payload = {"audio_b64": audio_b64, "format": "mp3"}
                log(f"Job complete. inline mp3 ({size} bytes)")
            else:
                log(f"MP3 is {size} bytes - uploading to Supabase...")
                url = upload_audio(mp3_path)
                out_payload = {"audio_url": url, "format": "mp3"}
                log(f"Job complete. url={url}")

        return out_payload

    except Exception as e:
        log("Job failed")
        traceback.print_exc()
        sys.stdout.flush()
        return {"error": str(e)}


log("Starting RunPod serverless loop...")
runpod.serverless.start({"handler": handler})
