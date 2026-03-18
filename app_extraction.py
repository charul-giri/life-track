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

# --- ADDED FOR TASK EXTRACTION ---
from transformers import pipeline
import re
# -----------------------------

# --- ADDED FOR GPU SUPPORT ---
import torch
# -----------------------------

# Print which ffmpeg is being used (useful for debugging)
print(shutil.which("ffmpeg"))

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------- CONFIGURATION ---------------- #

# Use Whisper "small" by default for speed; can override with env var WHISPER_MODEL
# Valid faster-whisper model names include: "tiny", "base", "small", "medium",
# "large-v2", "large-v3", etc.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
print(torch.cuda.is_available())

SAVE_FILE = os.path.join(os.getcwd(), "transcriptions.txt")
FFMPEG_PATH_ENV = os.environ.get("FFMPEG_PATH")  # optional: user can set this

if FFMPEG_PATH_ENV:
    # Append custom ffmpeg path to PATH
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH_ENV

# Select device: GPU if available, otherwise CPU
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
        "-y",                # overwrite
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        "-sample_fmt", "s16",
        output_path,
    ]

    # Use subprocess.run, check=True will raise an error if ffmpeg fails
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ---------------- MODEL LOADING ---------------- #

# Load Whisper model on the selected device (GPU/CPU) using faster-whisper
print(f"[INFO] Loading faster-whisper model: {WHISPER_MODEL} (this is much faster)...")

# Choose quantization type (speed vs accuracy)
# - On GPU: "int8_float16"
# - On CPU: "int8"
compute_type = "int8_float16" if DEVICE == "cuda" else "int8"

model = WhisperModel(
    WHISPER_MODEL,       # e.g. "small", "medium", "large-v2", "large-v3"
    device=DEVICE,       # "cuda" or "cpu"
    compute_type=compute_type,
)

print("[INFO] faster-whisper model loaded.")

# Load Task summarizer model (FLAN-T5) and move it to GPU if available
print("[INFO] Loading Task summarizer model (flan-t5-base)...")

# transformers pipeline uses:
#   device = -1  -> CPU
#   device = 0   -> first CUDA GPU
device_index = 0 if DEVICE == "cuda" else -1

task_summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    device=device_index,
)

print("[INFO] Task summarizer model loaded.")

# ------------- TASK EXTRACTION LOGIC ------------- #

def extract_tasks(conversation: str):
    """
    Extracts tasks from a conversation using the loaded FLAN-T5 pipeline.
    Returns a list of clean task strings.
    """
    prompt = (
        "Review the following conversation. Identify ONLY the specific tasks, goals, or objectives "
        "that are mentioned. Ignore greetings, pleasantries, and general conversation. "
        "If no tasks are mentioned, return 'None'. "
        "Extract the tasks as a clear, concise bullet list. "
        "Each bullet point must be a single, separate task.\n\n"
        f"Conversation:\n{conversation}"
    )

    try:
        result = task_summarizer(
            prompt,
            max_length=200,
            min_length=5,
            do_sample=False,
        )

        # Handle different Hugging Face pipeline outputs
        if isinstance(result, list):
            if "summary_text" in result[0]:
                result = result[0]["summary_text"]
            elif "generated_text" in result[0]:
                result = result[0]["generated_text"]
            else:
                result = str(result[0])

        if isinstance(result, str) and result.strip().lower() == "none":
            return []

    except Exception as e:
        print(f"[ERROR] Task summarizer failed: {e}")
        return []  # Return empty list on failure

    # Clean model output to a list of task strings
    def clean_to_list(summary: str):
        # Split by newlines, periods, commas, and 'and'
        lines = re.split(r"\.\s+|\n+|,|\s+and\s+", summary.strip())
        seen = set()
        cleaned = []
        for line in lines:
            line = line.strip(" -•").strip()
            # Basic sanity check so we only keep meaningful tasks
            if line and len(line) > 3:
                key = line.lower()
                if key not in seen:
                    seen.add(key)
                    cleaned.append(line)
        return cleaned

    return clean_to_list(result)

# ---------------- FLASK ROUTES ---------------- #

@app.route("/")
def index():
    # Serves your dashboard UI
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

        # data_url is like: "data:audio/webm;codecs=opus;base64,AAAA..."
        header, b64 = data_url.split(",", 1)
        audio_bytes = base64.b64decode(b64)

        # Save incoming blob to temp file
        fd_in, tmp_in = tempfile.mkstemp(suffix=".blob")
        os.close(fd_in)
        with open(tmp_in, "wb") as f:
            f.write(audio_bytes)

        # Convert to wav using ffmpeg
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

        # 1. Transcribe with faster-whisper (on GPU if available)
        segments, info = model.transcribe(
            tmp_wav,
            beam_size=5,   # reduce to 1–2 for more speed if needed
            language=None,
            task="transcribe",
        )

        # Join all segments into a single text string
        text = "".join(seg.text for seg in segments).strip()

        # 2. Extract Tasks
        tasks = []
        if text:
            tasks = extract_tasks(text)

        # 3. Save transcription to file with timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(SAVE_FILE, "a", encoding="utf-8") as out_f:
            out_f.write(f"[{timestamp}] {text}\n")

        # Return BOTH transcription and extracted tasks
        return jsonify({"text": text, "tasks": tasks})

    except Exception as e:
        print(f"[ERROR] An unhandled exception occurred: {e}")
        return jsonify({"error": "exception", "message": str(e)}), 500

    finally:
        # Clean up temp files
        try:
            if "tmp_in" in locals() and os.path.exists(tmp_in):
                os.remove(tmp_in)
            if "tmp_wav" in locals() and os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except Exception:
            # Don't crash just because cleanup failed
            pass


if __name__ == "__main__":
    # Runs on 0.0.0.0:5050 with debug enabled, same as before
    app.run(host="0.0.0.0", port=5050, debug=True)
