import os
import base64
import subprocess
import tempfile
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace-with-a-strong-secret")
CORS(app)

# ---------- Pages ----------
@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/call')
def call_page():
    return render_template('call.html')

# ---------- Session user info ----------
@app.route('/save_user_info', methods=['POST'])
def save_user_info():
    data = request.get_json() or {}
    # Basic validation
    name = data.get("name", "").strip()
    age = data.get("age", "").strip()
    gender = data.get("gender", "").strip()
    if not name or not age or not gender:
        return jsonify({"status": "error", "message": "Name, age, gender required"}), 400
    session['user_info'] = data
    print("Saved user_info:", data)
    return jsonify({"status": "ok", "saved": data})

@app.route('/get_user_info', methods=['GET'])
def get_user_info():
    return jsonify(session.get('user_info', {}))

# ---------- Chatbot processing (text) ----------
@app.route('/process_text', methods=['POST'])
def process_text():
    # Replace stub with emotion detection + LLM later
    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    user_info = session.get('user_info', {})
    # Example contextual reply: include user's name if provided
    name = user_info.get('name') if isinstance(user_info, dict) else None
    if name:
        reply = f"{name}, I heard you: {user_message}"
    else:
        reply = f"I heard you: {user_message}"
    print("process_text ->", {"user_message": user_message, "reply": reply})
    return jsonify({"reply": reply})

# ---------- TTS endpoint (gTTS -> mp3 -> ffmpeg -> wav) ----------
@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        wav_bytes = generate_tts_audio_ffmpeg(text)
        b64 = base64.b64encode(wav_bytes).decode('utf-8')
        return jsonify({"audio_base64": b64})
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:", e)
        return jsonify({"error": "ffmpeg conversion failed"}), 500
    except Exception as e:
        print("TTS error:", e)
        return jsonify({"error": "TTS generation failed"}), 500

def generate_tts_audio_ffmpeg(text: str) -> bytes:
    """
    Generate WAV bytes from gTTS via ffmpeg using temporary files.
    Works on Python 3.13 without pydub/audioop.
    Requires ffmpeg on PATH.
    """
    # create temp files
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(mp3_fd)
    os.close(wav_fd)
    try:
        # Write mp3 using gTTS
        tts = gTTS(text=text, lang="en")
        tts.save(mp3_path)

        # Use ffmpeg (must be installed & in PATH) to convert mp3 -> wav (PCM)
        # -y overwrite, -loglevel error to keep output clean
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_path,
            "-ar", "22050",   # sample rate (you can adjust)
            "-ac", "1",       # mono
            wav_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # read wav bytes
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        return wav_bytes
    finally:
        # cleanup temp files if they exist
        try:
            os.remove(mp3_path)
        except Exception:
            pass
        try:
            os.remove(wav_path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
