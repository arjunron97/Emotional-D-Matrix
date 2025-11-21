# app.py
import os
import uuid
import base64
import subprocess
import tempfile
import traceback

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()

try:
    from ron_service import RonService, DEFAULTS
except Exception as e:
    raise RuntimeError(f"Failed to import ron_service.py: {e}")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET", "replace-with-a-strong-secret")
CORS(app)

def init_ron_service():
    openai_key = os.environ.get("OPENAI_API_KEY")
    cfg = {
        "openai_api_key": openai_key,
        "chroma_dir": os.environ.get("CHROMA_DIR", DEFAULTS["CHROMA_DIR"]),
        "collection_name": os.environ.get("COLLECTION_NAME", DEFAULTS["COLLECTION_NAME"]),
        "cleaned_csv_path": os.environ.get("CLEANED_CSV_PATH", DEFAULTS["CLEANED_CSV_PATH"]),
        "embed_model_name": os.environ.get("EMBED_MODEL_NAME", DEFAULTS["EMBED_MODEL_NAME"]),
        "openai_chat_model": os.environ.get("OPENAI_CHAT_MODEL", DEFAULTS["OPENAI_CHAT_MODEL"]),
        "recreate_if_dim_mismatch": os.environ.get("RECREATE_IF_DIM_MISMATCH", str(DEFAULTS["RECREATE_IF_DIM_MISMATCH"])) == "True"
    }
    return RonService(**cfg)

try:
    RON_SERVICE = init_ron_service()
except Exception as e:
    print("[app] RonService init failed:", e)
    traceback.print_exc()
    RON_SERVICE = None

def ensure_session_id():
    sid = session.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
        session["session_id"] = sid
    return sid

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/call")
def call_page():
    return render_template("call.html")

@app.route("/start_chat", methods=["POST"])
def start_chat():
    if RON_SERVICE is None:
        return jsonify({"error": "service_unavailable"}), 503
    sid = ensure_session_id()
    user_info = request.get_json() or {}
    try:
        resp = RON_SERVICE.start_chat(sid, user_info=user_info)
        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "start_failed", "detail": str(e)}), 500

@app.route("/save_user_info", methods=["POST"])
def save_user_info():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    age = data.get("age", "").strip()
    gender = data.get("gender", "").strip()
    if not name or not age or not gender:
        return jsonify({"status": "error", "message": "Name, age, gender required"}), 400
    sid = ensure_session_id()
    try:
        if RON_SERVICE:
            RON_SERVICE.save_user_info(sid, data)
        session["user_info"] = data
        return jsonify({"status": "ok", "saved": data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "save_failed", "detail": str(e)}), 500

@app.route("/get_user_info", methods=["GET"])
def get_user_info():
    return jsonify(session.get("user_info", {}))

@app.route("/process_text", methods=["POST"])
def process_text():
    if RON_SERVICE is None:
        return jsonify({"error": "service_unavailable"}), 503
    sid = ensure_session_id()
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "no_message"}), 400
    try:
        result = RON_SERVICE.ron_reply(sid, user_message)
        return jsonify({
            "reply": result.get("reply"),
            "mode": result.get("mode"),
            "debug": result.get("debug"),
            "ended": result.get("ended", False)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "internal_error", "detail": str(e)}), 500

@app.route("/end_chat", methods=["POST"])
def end_chat():
    if RON_SERVICE is None:
        return jsonify({"error": "service_unavailable"}), 503
    sid = ensure_session_id()
    try:
        RON_SERVICE.mark_conversation_ended(sid)
        return jsonify({"status": "ok"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "end_failed", "detail": str(e)}), 500

@app.route("/rating")
def rating_page():
    return render_template("rating.html")

@app.route("/save_rating", methods=["POST"])
def save_rating():
    if RON_SERVICE is None:
        return jsonify({"error": "service_unavailable"}), 503
    sid = ensure_session_id()
    data = request.get_json() or {}
    rating = data.get("rating")
    comment = data.get("comment", "")
    try:
        rating_int = int(rating)
    except Exception:
        return jsonify({"error": "invalid_rating"}), 400
    ok = RON_SERVICE.save_rating(sid, rating_int, comment)
    if ok:
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "save_failed"}), 500

# ---------- TTS endpoint (default: gTTS -> mp3 -> ffmpeg -> wav) ----------
# NOTE: gTTS does not support gender selection. Use OpenAI TTS or Google Cloud TTS for male voices.
@app.route("/tts", methods=["POST"])
def tts():
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "no_text"}), 400

    try:
        # Male voice: alloy / verse
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="verse",    # ⬅⬅ male voice here (use "verse" for softer tone)
            input=text
        )

        audio_bytes = response.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return jsonify({ "audio_base64": audio_b64 })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "tts_failed", "detail": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    info = {"ok": False, "service": False}
    try:
        if RON_SERVICE is None:
            return jsonify(info), 503
        info["service"] = True
        info["collection_count"] = RON_SERVICE.collection_count()
        info["embeddings_exist"] = RON_SERVICE.embeddings_exist()
        info["ok"] = True
        return jsonify(info)
    except Exception as e:
        traceback.print_exc()
        info["ok"] = False
        info["error"] = str(e)
        return jsonify(info), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
