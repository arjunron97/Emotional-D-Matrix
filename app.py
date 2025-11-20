# app.py
import os
import base64
import subprocess
import tempfile
import traceback
from pathlib import Path

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from gtts import gTTS

# load environment variables from .env if present
load_dotenv()

# Import RonService and DEFAULTS
try:
    from ron_service import RonService, DEFAULTS
except Exception as e:
    # If import fails, raise early with a clear message
    raise RuntimeError(f"Failed to import ron_service: {e}")

# Flask app setup
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "replace-with-a-strong-secret")
CORS(app)

# Initialize RonService (one-time). Read config from environment with sensible defaults.
def init_ron_service():
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("[app] WARNING: OPENAI_API_KEY not found in env. RonService will raise on init.")
    cfg = {
        "openai_api_key": openai_key,
        "chroma_dir": os.environ.get("CHROMA_DIR", DEFAULTS["CHROMA_DIR"]),
        "collection_name": os.environ.get("COLLECTION_NAME", DEFAULTS["COLLECTION_NAME"]),
        "cleaned_csv_path": os.environ.get("CLEANED_CSV_PATH", DEFAULTS["CLEANED_CSV_PATH"]),
        "embed_model_name": os.environ.get("EMBED_MODEL_NAME", DEFAULTS["EMBED_MODEL_NAME"]),
        "openai_chat_model": os.environ.get("OPENAI_CHAT_MODEL", DEFAULTS["OPENAI_CHAT_MODEL"]),
        "recreate_if_dim_mismatch": os.environ.get("RECREATE_IF_DIM_MISMATCH", "True") == "True"
    }
    print("[app] Initializing RonService with config:", {k: v for k, v in cfg.items() if k != "openai_api_key"})
    return RonService(**cfg)

# Create a globally available RonService instance
try:
    RON_SERVICE = init_ron_service()
except Exception as e:
    # If RonService fails to init, keep app running but mark service as None
    print("[app] Failed to initialize RonService:", e)
    traceback.print_exc()
    RON_SERVICE = None

# ---------- Pages ----------
@app.route("/")
def landing():
    # Expect index.html in templates/
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/call")
def call_page():
    return render_template("call.html")

# ---------- Session user info ----------
@app.route("/save_user_info", methods=["POST"])
def save_user_info():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    age = data.get("age", "").strip()
    gender = data.get("gender", "").strip()
    if not name or not age or not gender:
        return jsonify({"status": "error", "message": "Name, age, gender required"}), 400
    session["user_info"] = data
    print("[app] Saved user_info:", data)
    return jsonify({"status": "ok", "saved": data})

@app.route("/get_user_info", methods=["GET"])
def get_user_info():
    return jsonify(session.get("user_info", {}))

# ---------- Chatbot processing (text) ----------
@app.route("/process_text", methods=["POST"])
def process_text():
    if RON_SERVICE is None:
        return jsonify({"error": "service_unavailable", "message": "RonService not initialized"}), 503

    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    user_info = session.get("user_info", {})

    if not user_message:
        return jsonify({"error": "no_message", "message": "No message provided"}), 400

    try:
        result = RON_SERVICE.ron_reply(user_message)
        reply_text = result.get("reply", "")
        # Optionally personalize or inject user_info into reply (kept minimal here)
        return jsonify({
            "reply": reply_text,
            "mode": result.get("mode"),
            "debug": result.get("debug")   # remove this in production for privacy
        })
    except Exception as e:
        print("[app] Error in /process_text:", e)
        traceback.print_exc()
        return jsonify({"error": "internal_error", "detail": str(e)}), 500

# ---------- TTS endpoint (gTTS -> ffmpeg -> wav bytes -> base64) ----------
@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "no_text", "message": "No text provided"}), 400
    try:
        wav_bytes = generate_tts_audio_ffmpeg(text)
        b64 = base64.b64encode(wav_bytes).decode("utf-8")
        return jsonify({"audio_base64": b64})
    except subprocess.CalledProcessError as e:
        print("[app] ffmpeg failed:", e)
        return jsonify({"error": "ffmpeg_failed", "detail": str(e)}), 500
    except Exception as e:
        print("[app] TTS generation failed:", e)
        traceback.print_exc()
        return jsonify({"error": "tts_failed", "detail": str(e)}), 500

def generate_tts_audio_ffmpeg(text: str) -> bytes:
    """
    Generate WAV bytes from gTTS using ffmpeg conversion.
    Requires ffmpeg to be installed & available on PATH.
    """
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(mp3_fd)
    os.close(wav_fd)
    try:
        tts = gTTS(text=text, lang="en")
        tts.save(mp3_path)
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", "22050",
            "-ac", "1",
            wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes
    finally:
        try:
            os.remove(mp3_path)
        except Exception:
            pass
        try:
            os.remove(wav_path)
        except Exception:
            pass

# ---------- Health endpoint ----------
@app.route("/health", methods=["GET"])
def health():
    info = {"ok": False, "service": False, "collection_count": None, "embeddings_exist": None}
    try:
        if RON_SERVICE is None:
            info["ok"] = False
            info["service"] = False
            return jsonify(info), 503
        info["service"] = True
        info["collection_count"] = RON_SERVICE.collection_count()
        info["embeddings_exist"] = RON_SERVICE.embeddings_exist()
        info["ok"] = True
        return jsonify(info)
    except Exception as e:
        print("[app] Health check error:", e)
        traceback.print_exc()
        info["ok"] = False
        info["error"] = str(e)
        return jsonify(info), 500

if __name__ == "__main__":
    # Run in debug during development only
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
