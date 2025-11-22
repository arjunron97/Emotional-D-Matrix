# Emotional-D-Matrix - RON
 **Ron — your warm AI emotional support companion**  
> A retrieval-augmented conversational assistant that listens, validates feelings, and offers short, empathic coping steps. Built with an LLM + vector retrieval over the GoEmotions dataset, Chroma vector store, and a lightweight Flask frontend (text + voice/call mode).

---

**Project Presentation ppt link** - https://myunt-my.sharepoint.com/:p:/g/personal/arjunpalaniswamy_my_unt_edu/EbbOPLy6QAJHlTFEVTEQBrkBPOlgC5nuh2QSjp6Qh3vPPw?e=HcrisL

**Project Presentation Video Link** - https://teams.microsoft.com/l/meetingrecap?driveId=b%21hKGJcgUEx0Seo_fbaEXNqmJ7HViYYRVGlOpoPHYQDb3noTAHgfWfSZgYoNbl78QB&driveItemId=01WMPJLFFIEEVH2LVPTBHILQEOVT6FNWQB&sitePath=https%3A%2F%2Fmyunt-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Farjunpalaniswamy_my_unt_edu%2FEaghKn0ur5hOhcCOrPxW2gEB49QoGGzhV2pD5TD0vtyEXw&fileUrl=https%3A%2F%2Fmyunt-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Farjunpalaniswamy_my_unt_edu%2FEaghKn0ur5hOhcCOrPxW2gEB49QoGGzhV2pD5TD0vtyEXw&threadId=19%3Ameeting_YWRhOTVjMmEtNDBiOC00MjI2LWI2NjEtNzJkYmU5NTc0OGE5%40thread.v2&organizerId=1147f8a4-6e50-4f92-94ee-0451f19b7e57&tenantId=70de1992-07c6-480f-a318-a1afcba03983&callId=0ec19c7a-2d5a-4746-a497-25f833a955b7&threadType=Meeting&meetingType=MeetNow&subType=RecapSharingLink_RecapChiclet 

**Project Document Link** - https://myunt-my.sharepoint.com/:w:/g/personal/arjunpalaniswamy_my_unt_edu/EUoSxYt0v8hIqAttHeWbv5wBziAFBp9wWSwj_5hCW-DWTA?e=vLlJtV

**Survy Link** - https://forms.cloud.microsoft/r/ft0knsJYhv

---

## 🔎 About this project
**Emotional D-Matrix** is a prototype conversational system whose persona is **Ron** — a friendly, human-like emotional support assistant.  
Ron combines semantic retrieval from a labeled emotion dataset (GoEmotions) with an LLM to produce grounded, empathic replies. The system supports both typed chat and a voice/call mode with waveform visualization and TTS.

This repository demonstrates:
- building a vector DB from GoEmotions,
- retrieval-augmented generation (RAG) to inform responses,
- a conversational system prompt tuned for empathy,
- frontend voice capabilities (Web Speech API + server TTS),
- persistent conversation and feedback storage (SQLite).

> **Important:** This is a supportive prototype, **not** a clinical or crisis intervention tool. See **Ethics & Safety** below.

---

## 🧾 Abstract
People often need someone to listen. This project shows how to combine embeddings, vector search, and an LLM to build an assistant that:
1. Detects whether user text is a problem statement,  
2. Retrieves similar emotional examples (GoEmotions) from a vector store (Chroma),  
3. Uses those retrievals to ground a brief, empathetic LLM reply, and  
4. Provides a voice experience (live input visualization + TTS) with conversation persistence and post-chat rating.

---

## 🚀 Features
- Warm persona: Ron introduces himself and asks a follow-up.
- Problem detection: heuristics decide when to retrieve emotion labels/examples.
- Semantic retrieval: Chroma + sentence-transformers embeddings over GoEmotions.
- Contextual chat: uses recent turns for coherent replies.
- Voice/call mode: live waveform, mic input, output playback (TTS).
- Persists users, messages, conversations, and ratings in SQLite for analysis.
- Safety heuristics: basic crisis detection with a signpost to emergency resources.

---

## 🛠️ Who is this useful for?
- Researchers and students learning RAG for emotional/NLP tasks.
- Developers building prototypes combining embeddings + LLMs.
- Educators demonstrating conversational UX and safety considerations.
- Hobbyists exploring voice + retrieval-augmented chatbots.

_Not suitable as a replacement for licensed mental health support._

---

## 🧩 Project structure (high level)
```
.
├─ app.py                 # Flask app + endpoints (start_chat, process_text, tts, rating)
├─ ron_service.py         # Core logic: embedder, Chroma, retrieval, LLM calls, sqlite storage
├─ ron_test.py            # Integration tests & quick checks
├─ templates/
│  ├─ call.html
│  └─ rating.html
├─ static/
│  ├─ js/call_tts.js
│  └─ css/style.css
├─ notebook/
│  └─ dataset/
│     └─ goemotions_cleaned.csv
├─ chroma_db/             # Chroma persistent directory (created at runtime)
├─ chat_data.db           # SQLite (created at runtime)
└─ README.md
```

---

## ✅ Prerequisites
- Python 3.9+ (3.10/3.11 recommended)
- Internet access (to download sentence-transformers & for OpenAI calls)
- (Optional) `ffmpeg` — only if you use local gTTS fallback for TTS
- An OpenAI API key (recommended for chat + TTS)

---

## 🧰 Install & setup

1. Clone repo and create virtualenv
```bash
git clone <your-repo-url>
cd Emotional-D-Matrix
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies (example list)
```bash
pip install --upgrade pip
pip install flask flask-cors python-dotenv requests pandas ftfy tqdm sentence-transformers chromadb gTTS
```
> If you use OpenAI TTS (recommended), also install the official client:
```bash
pip install openai
```

3. Create `.env` in project root (example):
```
OPENAI_API_KEY=sk-...
FLASK_SECRET=replace-with-a-strong-secret
CHROMA_DIR=./chroma_db
CLEANED_CSV_PATH=notebook/dataset/goemotions_cleaned.csv
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
OPENAI_CHAT_MODEL=gpt-3.5-turbo
TTS_VOICE=verse           # optional if using OpenAI TTS ("verse" or "alloy")
```
> **Keep .env out of version control.**

---

## ▶️ Run (development)
```bash
# activate venv
python app.py
# visit:
# http://localhost:5000/chat   (text UI)
# http://localhost:5000/call   (voice/call UI)
```

---

## 🧪 Testing
Run the integration checks:
```bash
python ron_test.py
```
This script:
- initializes `RonService`,
- calls `start_chat`,
- runs a free-chat sample,
- runs a problem statement sample,
- simulates satisfaction & checks DB entries.

---

## ⚙️ Embeddings & Chroma
- Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
- On first run RonService will check `CHROMA_DIR` collection:
  - If empty and `CLEANED_CSV_PATH` exists, it will create embeddings and populate Chroma.
  - If embeddings already exist, they will be reused. **No re-embedding required** unless you change the model.
- If you change the embedding model and dimensions mismatch, delete `chroma_db` or enable `RECREATE_IF_DIM_MISMATCH`.

---

## 🗂 API (for frontend)
- `POST /start_chat` → `{ reply, mode }`
- `POST /process_text` → `{ reply, mode, debug, ended }` (`ended: true` means conversation satisfied)
- `POST /end_chat` → mark conversation ended
- `POST /save_user_info` → save name/age/gender
- `POST /tts` → `{ audio_base64 }` (returns WAV/MP3 bytes as base64)
- `GET /rating` → rating page
- `POST /save_rating` → save numeric rating + comment
- `GET /health` → service status

---

## 💡 UX & Behavior notes
- Ron introduces himself automatically on chat start with a warm, humanlike intro.
- The model prompt is tuned so Ron **does not prefix answers with "Ron:"**.
- Short follow-ups are encouraged — Ron asks gentle questions rather than lengthy lectures.
- Conversation context: last `N` turns are provided to the model to maintain coherence.

---

## 🔐 Data, privacy & ethics
- Conversations & ratings are saved to SQLite (`chat_data.db`) for analysis.
- **Do not** collect or store sensitive personal information unless you have explicit user consent.
- Add a visible disclaimer in UI: this bot is **not** a replacement for professional help.
- Provide crisis resources (e.g., **988** for U.S.) wherever appropriate.

---

## 🛠 Troubleshooting (common issues)
- **Chroma dimension mismatch**: Happens if embedding model changed. Delete `chroma_db` and rebuild, or set `RECREATE_IF_DIM_MISMATCH=True`.
- **OpenAI errors**: Check `OPENAI_API_KEY` and network access.
- **TTS**: If using gTTS + ffmpeg ensure `ffmpeg` is installed and in PATH. Prefer OpenAI TTS for higher quality and male voice selection.
- **Browser SpeechRecognition**: Web Speech API is Chrome-friendly. For cross-browser ASR, integrate server-side Whisper or cloud ASR.

---

## 🚧 Limitations
- Not a clinical tool. No diagnosis, no therapy.
- Heuristic detection may miss or misclassify some problem statements.
- Current safety handling is signposting — for production you must integrate a robust crisis management flow.

---

## 🔭 Roadmap / Future ideas
- Add opt-in conversational memory for follow-ups.
- Server-side ASR (Whisper) for higher accuracy.
- Multi-voice presets + voice selector in UI.
- Analytics pipeline for emotion trends and conversation quality.
- Fine-tune or safety-filter responses for consistent supportive tone.
- Integrate anonymization or PII removal before storing text.

---

## 🤝 Contributing
- Fork, add features and open PRs.
- Add unit tests for `ron_service` (heuristics, retrieval).
- Keep secrets out of commits.

---

## 📜 License
MIT — add a `LICENSE` file if you plan to open-source.

---

## 🎁 Extras (examples you can paste)
**.env snippet**
```env
OPENAI_API_KEY=sk-...
FLASK_SECRET=mysecret
CHROMA_DIR=./chroma_db
CLEANED_CSV_PATH=notebook/dataset/goemotions_cleaned.csv
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
OPENAI_CHAT_MODEL=gpt-3.5-turbo
TTS_VOICE=verse
```

**Quick commands**
```bash
# (re)build chroma from cleaned CSV
python -c "from ron_service import RonService; RonService(openai_api_key='sk-...')._build_collection_from_csv('notebook/dataset/goemotions_cleaned.csv', RonService().collection)"
```

---

## Final note
This project is intentionally modular — swap embedding models, LLMs, TTS, or the vector DB with minimal changes.  
If you want, I can:
- produce a `requirements.txt` and `setup.sh`,
- generate a privacy policy snippet and in-app disclaimer,
- create a short demo script / slides to present Ron.

Which of those would you like next?
