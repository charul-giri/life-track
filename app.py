# app.py
import os
import shutil
import tempfile
import base64
import subprocess
from datetime import datetime

from flask import Flask, render_template, request, jsonify

# --- WHISPER (FASTER VERSION) ---
from faster_whisper import WhisperModel
# -----------------------------

# --- ADDED: IMPORT FLUSH VERSION MODULE ---  # <<< CHANGED
from task_extractor import extract_tasks_from_paragraph  # make sure flush_version.py is in same folder
# -----------------------------

# --- ADDED FOR GPU SUPPORT ---
import torch
# -----------------------------

# Print which ffmpeg is being used (useful for debugging)
print(shutil.which("ffmpeg"))

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------- CONFIGURATION ---------------- #

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
print(torch.cuda.is_available())

SAVE_FILE = os.path.join(os.getcwd(), "transcriptions.txt")
FFMPEG_PATH_ENV = os.environ.get("FFMPEG_PATH")  # optional: user can set this

if FFMPEG_PATH_ENV:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH_ENV

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# ---------------- FFMPEG HELPERS ---------------- #

def ffmpeg_available() -> bool:
    """Check if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def convert_to_wav(input_path: str, output_path: str) -> None:
    """
    Use ffmpeg to convert arbitrary audio (webm/ogg/opus/etc.) to WAV 16k mono.
    Raises subprocess.CalledProcessError on failure.
    """
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise FileNotFoundError("ffmpeg executable not found on PATH.")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        "-sample_fmt", "s16",
        output_path,
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ---------------- MODEL LOADING ---------------- #

print(f"[INFO] Loading faster-whisper model: {WHISPER_MODEL} (this is much faster)...")

compute_type = "int8_float16" if DEVICE == "cuda" else "int8"

model = WhisperModel(
    WHISPER_MODEL,
    device=DEVICE,
    compute_type=compute_type,
)

print("[INFO] faster-whisper model loaded.")

# NOTE: Old FLAN-T5 task_summarizer and extract_tasks() have been REMOVED   # <<< REMOVED
# We now use flush_version.extract_tasks_from_paragraph() instead.

# ---------------- FLASK ROUTES ---------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if not ffmpeg_available():
        return jsonify(
            {
                "error": "ffmpeg_not_found",
                "message": (
                    "ffmpeg executable not found. Install ffmpeg and ensure it's on PATH, "
                    "or set environment variable FFMPEG_PATH. See: https://ffmpeg.org/download.html"
                ),
            }
        ), 500

    try:
        payload = request.get_json(force=True)
        data_url = payload.get("audio")
        if not data_url or "," not in data_url:
            return jsonify({"error": "invalid_payload", "message": "No audio data provided"}), 400

        # data_url: "data:audio/webm;codecs=opus;base64,AAAA..."
        header, b64 = data_url.split(",", 1)
        audio_bytes = base64.b64decode(b64)

        # temp input file
        fd_in, tmp_in = tempfile.mkstemp(suffix=".blob")
        os.close(fd_in)
        with open(tmp_in, "wb") as f:
            f.write(audio_bytes)

        # convert to wav
        fd_wav, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd_wav)

        try:
            convert_to_wav(tmp_in, tmp_wav)
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", "ignore")
            print(f"[ERROR] ffmpeg conversion failed: {err_msg}")
            raise RuntimeError(f"ffmpeg convert failed: {err_msg}")
        except FileNotFoundError as e:
            print(f"[ERROR] ffmpeg not found: {e}")
            raise RuntimeError("ffmpeg executable not found, check PATH.")

        # 1. Transcribe with faster-whisper
        segments, info = model.transcribe(
            tmp_wav,
            beam_size=5,
            language=None,
            task="transcribe",
        )

        text = "".join(seg.text for seg in segments).strip()

        # 2. NEW: Use flush_version to extract tasks + times   # <<< CHANGED
        tasks = []
        if text:
            tasks = extract_tasks_from_paragraph(text)

        # 3. Save transcription
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(SAVE_FILE, "a", encoding="utf-8") as out_f:
            out_f.write(f"[{timestamp}] {text}\n")

        # Return transcription + structured tasks
        return jsonify({"text": text, "tasks": tasks})

    except Exception as e:
        print(f"[ERROR] An unhandled exception occurred: {e}")
        return jsonify({"error": "exception", "message": str(e)}), 500

    finally:
        try:
            if "tmp_in" in locals() and os.path.exists(tmp_in):
                os.remove(tmp_in)
            if "tmp_wav" in locals() and os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
