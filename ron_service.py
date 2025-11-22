# ron_service.py
import os
import re
import json
import time
import sqlite3
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime

import requests
import pandas as pd
import ftfy
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# -------- DEFAULTS --------
DEFAULTS = {
    "CHROMA_DIR": "./chroma_db",
    "COLLECTION_NAME": "goemotions_dataset",
    "CLEANED_CSV_PATH": "notebook/dataset/goemotions_cleaned.csv",
    "EMBED_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
    "EMBED_DIM_EXPECTED": 384,
    "OPENAI_CHAT_MODEL": "gpt-3.5-turbo",
    "RECREATE_IF_DIM_MISMATCH": False,
    "DEFAULT_TOP_K": 6,
    "REQUEST_TIMEOUT": 30.0,
    "MAX_RETRIES": 3,
    "RETRY_BACKOFF": 1.5,
    "OPENAI_API_BASE": "https://api.openai.com/v1",
    "SQLITE_PATH": "./chat_data.db",
    "CONTEXT_TURNS": 6
}

def clean_text(t: Optional[str]) -> str:
    if t is None:
        return ""
    t = str(t).strip()
    t = ftfy.fix_text(t)
    t = t.replace("\n", " ").replace("\r", " ")
    return " ".join(t.split())

def contains_any(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)

class RonService:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        chroma_dir: str = DEFAULTS["CHROMA_DIR"],
        collection_name: str = DEFAULTS["COLLECTION_NAME"],
        cleaned_csv_path: str = DEFAULTS["CLEANED_CSV_PATH"],
        embed_model_name: str = DEFAULTS["EMBED_MODEL_NAME"],
        embed_dim_expected: int = DEFAULTS["EMBED_DIM_EXPECTED"],
        openai_chat_model: str = DEFAULTS["OPENAI_CHAT_MODEL"],
        recreate_if_dim_mismatch: bool = DEFAULTS["RECREATE_IF_DIM_MISMATCH"],
        default_top_k: int = DEFAULTS["DEFAULT_TOP_K"],
        request_timeout: float = DEFAULTS["REQUEST_TIMEOUT"],
        max_retries: int = DEFAULTS["MAX_RETRIES"],
        retry_backoff: float = DEFAULTS["RETRY_BACKOFF"],
        openai_api_base: str = DEFAULTS["OPENAI_API_BASE"],
        sqlite_path: str = DEFAULTS["SQLITE_PATH"],
        context_turns: int = DEFAULTS["CONTEXT_TURNS"]
    ):
        # Config + API key
        self.OPENAI_API_KEY = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.OPENAI_CHAT_MODEL = openai_chat_model
        self.OPENAI_API_BASE = openai_api_base
        self.REQUEST_TIMEOUT = request_timeout
        self.MAX_RETRIES = max_retries
        self.RETRY_BACKOFF = retry_backoff

        self.CHROMA_DIR = chroma_dir
        self.COLLECTION_NAME = collection_name
        self.CLEANED_CSV_PATH = cleaned_csv_path
        self.EMBED_MODEL_NAME = embed_model_name
        self.EMBED_DIM_EXPECTED = embed_dim_expected
        self.RECREATE_IF_DIM_MISMATCH = recreate_if_dim_mismatch
        self.DEFAULT_TOP_K = default_top_k
        self.SQLITE_PATH = sqlite_path
        self.CONTEXT_TURNS = context_turns

        # Load embedder
        print("[RonService] Loading embedder:", self.EMBED_MODEL_NAME)
        self.embedder = SentenceTransformer(self.EMBED_MODEL_NAME)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        print("[RonService] Embedder dim:", self.embed_dim)
        if self.embed_dim != self.EMBED_DIM_EXPECTED:
            print(f"[RonService] WARNING: embedder dim {self.embed_dim} != expected {self.EMBED_DIM_EXPECTED}")

        # Chroma client
        print("[RonService] Opening Chroma at:", self.CHROMA_DIR)
        self.client_db = PersistentClient(path=self.CHROMA_DIR)
        try:
            existing = self.client_db.list_collections()
            self.existing_names = [c.name for c in existing]
        except Exception as e:
            print("[RonService] Warning listing collections:", e)
            self.existing_names = []
        self.collection = None
        self._ensure_collection_ready()

        # REST headers for OpenAI (used by call_openai_chat_rest)
        self.HEADERS = {"Authorization": f"Bearer {self.OPENAI_API_KEY}", "Content-Type": "application/json"}

        # init sqlite and tables
        self._init_sqlite(self.SQLITE_PATH)

        # heuristics
        self.PROBLEM_KEYWORDS = [
            "problem", "issue", "need help", "help me", "i can't", "i cannot", "i don't know",
            "i'm stuck", "i am stuck", "confused", "scared", "depressed", "can't handle",
            "anxious", "panic", "panic attack", "breakdown", "stress", "stressed", "overwhelmed",
            "scolded", "fired", "quit", "abuse", "harassed"
        ]
        self.SAFETY_KEYWORDS = [
            "suicide","kill myself","want to die","end my life","hurting myself","commit suicide",
            "i will kill myself","i am going to kill myself","i want to die"
        ]
        self.SATISFACTION_KEYWORDS = [
          "that helps", "solved", "done", "perfect",  "bye", "goodbye"
        ]

        self.SYSTEM_PROMPT = (
            "You are Ron — a warm, friendly, conversational AI who provides emotional support. "
            "Speak like a caring human in short, natural turns. "
            "Do not start responses with the bot's name or 'Ron:'. "
            "Keep responses empathetic and conversational, ask brief follow-up questions, "
            "and avoid long lists or lecture-like advice. Use simple language and be encouraging."
        )

        print("[RonService] Ready.")

    # ---------------- SQLITE ----------------
    def _init_sqlite(self, path: str):
        self.sqlite_path = path
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            session_id TEXT PRIMARY KEY,
            user_info JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            session_id TEXT PRIMARY KEY,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended INTEGER DEFAULT 0
        )
        """)
        # ratings table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            rating INTEGER,
            comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        conn.close()

    def save_user_info(self, session_id: str, user_info: dict):
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO users (session_id, user_info) VALUES (?, ?)",
                    (session_id, json.dumps(user_info)))
        conn.commit()
        conn.close()

    def start_conversation_record(self, session_id: str):
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO conversations (session_id) VALUES (?)", (session_id,))
        conn.commit()
        conn.close()

    def save_message(self, session_id: str, role: str, text: str, ended: int = 0):
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("INSERT INTO messages (session_id, role, text, ended) VALUES (?, ?, ?, ?)",
                    (session_id, role, text, ended))
        conn.commit()
        conn.close()

    def mark_conversation_ended(self, session_id: str):
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("UPDATE conversations SET ended_at = CURRENT_TIMESTAMP WHERE session_id = ?", (session_id,))
        cur.execute("UPDATE messages SET ended = 1 WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    def save_rating(self, session_id: str, rating: int, comment: str = "") -> bool:
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cur = conn.cursor()
            cur.execute("INSERT INTO ratings (session_id, rating, comment) VALUES (?, ?, ?)",
                        (session_id, int(rating), comment[:2000]))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print("[RonService] save_rating error:", e)
            return False

    def get_recent_messages(self, session_id: str, n: int = None) -> List[Dict[str, Any]]:
        n = n or self.CONTEXT_TURNS
        conn = sqlite3.connect(self.sqlite_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT role, text, created_at FROM messages
            WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
        """, (session_id, n))
        rows = cur.fetchall()
        conn.close()
        rows = list(reversed(rows))
        return [{"role": r[0], "text": r[1], "created_at": r[2]} for r in rows]

    # ---------------- CHROMA helpers ----------------
    def _detect_collection_dim_mismatch(self, col_obj, sample_text: str = "hello"):
        try:
            emb = self.embedder.encode([sample_text], convert_to_numpy=True).tolist()
            col_obj.query(query_embeddings=emb, n_results=1)
            return False, None, self.embed_dim
        except Exception as e:
            msg = str(e)
            m = re.search(r"expecting embedding with dimension of (\d+), got (\d+)", msg)
            if m:
                expected = int(m.group(1)); got = int(m.group(2))
                return True, expected, got
            return True, None, self.embed_dim

    def _safe_recreate_collection(self):
        name = self.COLLECTION_NAME
        try:
            col = self.client_db.get_or_create_collection(name=name)
            try:
                col.delete()
            except Exception:
                pass
            try:
                self.client_db.delete_collection(name)
            except Exception:
                pass
            try:
                newcol = self.client_db.create_collection(name=name)
                return newcol
            except Exception:
                return self.client_db.get_or_create_collection(name=name)
        except Exception:
            return self.client_db.create_collection(name=name)

    def _ensure_collection_ready(self):
        name = self.COLLECTION_NAME
        if name in self.existing_names:
            try:
                self.collection = self.client_db.get_collection(name)
                mismatch, expected, got = self._detect_collection_dim_mismatch(self.collection)
                if mismatch and self.RECREATE_IF_DIM_MISMATCH:
                    self.collection = self._safe_recreate_collection()
            except Exception as e:
                print("[RonService] error loading collection:", e)
                self.collection = self._safe_recreate_collection()
        else:
            self.collection = self._safe_recreate_collection()

        try:
            cnt = 0
            try:
                cnt = self.collection.count()
            except Exception:
                cnt = 0
            if int(cnt) == 0:
                p = Path(self.CLEANED_CSV_PATH)
                if p.exists():
                    self._build_collection_from_csv(self.CLEANED_CSV_PATH, self.collection)
        except Exception:
            pass

    def _build_collection_from_csv(self, csv_path: str, col_obj, batch_size: int = 512):
        p = Path(csv_path)
        df = pd.read_csv(p, dtype=str, low_memory=False).fillna("")
        if "text" not in df.columns:
            return
        df['text'] = df['text'].apply(clean_text)
        if 'example_very_unclear' in df.columns:
            df['example_very_unclear'] = df['example_very_unclear'].astype(str).str.strip().str.upper().isin(['TRUE','T','1','YES','Y'])
            df = df[~df['example_very_unclear']]
        if 'id' in df.columns:
            df = df.drop_duplicates(subset=['id'])
        else:
            df = df.drop_duplicates(subset=['text'])
        df = df[df['text'].str.strip() != ""].reset_index(drop=True)
        n = len(df)
        if n == 0:
            return
        ignore_cols = {"text","id","author","subreddit","link_id","parent_id","created_utc","rater_id","example_very_unclear"}
        possible_label_cols = [c for c in df.columns if c not in ignore_cols]
        ids_all = [f"ex_{i}" for i in range(n)]
        for i in tqdm(range(0, n, batch_size), desc="Indexing batches"):
            batch = df.iloc[i:i+batch_size]
            texts = batch['text'].tolist()
            embs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            metas = []
            for _, row in batch.iterrows():
                positives = []
                for c in possible_label_cols:
                    try:
                        if int(str(row.get(c, "0")).strip()) == 1:
                            positives.append(c)
                    except Exception:
                        pass
                labels_str = ",".join(positives)
                label_count = len(positives)
                meta = {
                    "id": row.get("id", "") or "",
                    "author": row.get("author", "") or "",
                    "subreddit": row.get("subreddit", "") or "",
                    "created_utc": row.get("created_utc", "") or "",
                    "labels": labels_str,
                    "label_count": label_count
                }
                metas.append(meta)
            batch_ids = ids_all[i:i+batch_size]
            col_obj.add(ids=batch_ids, documents=texts, metadatas=metas, embeddings=embs.tolist())
        try:
            self.client_db.persist()
        except Exception:
            pass

    # ---------------- OpenAI REST wrapper ----------------
    def call_openai_chat_rest(self, messages: List[Dict[str, str]], model: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 350) -> str:
        model_name = model or self.OPENAI_CHAT_MODEL
        url = f"{self.OPENAI_API_BASE}/chat/completions"
        payload = {"model": model_name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        backoff = 1.0
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                r = requests.post(url, headers=self.HEADERS, json=payload, timeout=self.REQUEST_TIMEOUT)
            except requests.RequestException as e:
                if attempt == self.MAX_RETRIES:
                    return f"[Network error calling OpenAI: {e}]"
                time.sleep(backoff)
                backoff *= self.RETRY_BACKOFF
                continue
            if r.status_code == 200:
                try:
                    j = r.json()
                    choice = j.get("choices", [None])[0]
                    if not choice:
                        return "[OpenAI response missing choices]"
                    msg = choice.get("message") or {}
                    content = msg.get("content")
                    if content is None:
                        content = choice.get("text") or json.dumps(choice)
                    return content
                except Exception as e:
                    return f"[Error parsing OpenAI response: {e} | raw: {r.text}]"
            else:
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    if attempt == self.MAX_RETRIES:
                        try:
                            return f"[OpenAI API error {r.status_code}: {r.json()}]"
                        except Exception:
                            return f"[OpenAI API error {r.status_code}: {r.text}]"
                    time.sleep(backoff)
                    backoff *= self.RETRY_BACKOFF
                    continue
                else:
                    try:
                        return f"[OpenAI API error {r.status_code}: {r.json()}]"
                    except Exception:
                        return f"[OpenAI API error {r.status_code}: {r.text}]"
        return "[OpenAI call failed after retries]"

    # ---------------- retrieval + inference ----------------
    def infer_emotions_from_text(self, user_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        top_k = top_k or self.DEFAULT_TOP_K
        emb = self.embedder.encode([user_text], convert_to_numpy=True).tolist()
        try:
            res = self.collection.query(query_embeddings=emb, n_results=top_k, include=["metadatas","documents","distances"])
        except Exception as e:
            return {"inferred": [], "neighbors": [], "error": str(e)}
        metas = res.get("metadatas", [])
        docs = res.get("documents", [])
        dists = res.get("distances", [])
        if not metas or not docs:
            return {"inferred": [], "neighbors": [], "error": "empty_result"}
        meta_list = metas[0]
        docs_list = docs[0]
        dist_list = dists[0]
        label_counter = Counter()
        for m in meta_list:
            labels_str = m.get("labels","")
            labs = [s for s in labels_str.split(",") if s]
            for lab in labs:
                label_counter[lab] += 1
        total = sum(label_counter.values()) or 1
        inferred = [{"label": lab, "count": cnt, "confidence": cnt/total} for lab,cnt in label_counter.most_common(3)]
        neighbors = []
        for d,m,dist in zip(docs_list, meta_list, dist_list):
            neighbors.append({"text": d, "meta": m, "distance": dist})
        return {"inferred": inferred, "neighbors": neighbors, "error": None}

    # ---------------- build messages with context ----------------
    def _build_messages_with_context(self, session_id: str, user_text: str, problem_mode: bool, info: Optional[dict] = None) -> List[Dict[str,str]]:
        history = self.get_recent_messages(session_id, n=self.CONTEXT_TURNS)
        conversation_messages = []
        for turn in history:
            conversation_messages.append({"role": turn["role"], "content": turn["text"]})
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(conversation_messages)
        if problem_mode and info is not None:
            inferred = info.get("inferred", [])
            inferred_str = ", ".join([f"{x['label']}({x['confidence']:.2f})" for x in inferred]) if inferred else "unknown"
            neighbors = info.get("neighbors", [])
            examples = "\n".join([f"- \"{n['text'][:140]}\" labels:{n['meta'].get('labels')} dist={n['distance']:.4f}" for n in neighbors[:3]]) or "none"
            guide = (
                f"User: {user_text}\n"
                f"Inferred emotions: {inferred_str}\n"
                f"Relevant examples:\n{examples}\n\n"
                "Respond conversationally: validate briefly, offer 1 immediate coping action, 1 short medium-term suggestion, and ask a brief follow-up question."
            )
            messages.append({"role": "user", "content": guide})
        else:
            messages.append({"role": "user", "content": user_text})
        return messages

    # ---------------- heuristics ----------------
    def is_problem_statement(self, text: str) -> bool:
        txt = (text or "").lower()
        if re.search(r"\b(help|advice|what should i do|how do i|should i)\b", txt):
            return True
        if contains_any(txt, self.PROBLEM_KEYWORDS):
            return True
        if len(txt.split()) > 6 and re.search(r"\b(i am|i'm|i feel|i was|i got)\b", txt) and contains_any(txt, ["sad","angry","upset","embarrassed","ashamed","afraid","scared","anxious","depressed","stressed","hurt"]):
            return True
        return False

    def is_safety_risk(self, text: str) -> bool:
        return contains_any((text or "").lower(), self.SAFETY_KEYWORDS)

    def is_user_satisfied(self, text: str) -> bool:
        txt = (text or "").lower()
        return contains_any(txt, self.SATISFACTION_KEYWORDS)

    # ---------------- public API ----------------
    def start_chat(self, session_id: str, user_info: Optional[dict] = None) -> Dict[str, Any]:
        session_id = session_id or "anonymous"
        self.start_conversation_record(session_id)
        if user_info:
            self.save_user_info(session_id, user_info)
        intro = (
            "Hey there! I'm Ron — your AI emotional support companion. "
            "I'm here to listen, understand, and help you feel a little lighter today. "
            "So tell me… how’s your day going so far?"
        )
        self.save_message(session_id, "assistant", intro)
        return {"reply": intro, "mode": "intro", "debug": None}

    def ron_reply(self, session_id: str, user_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        session_id = session_id or "anonymous"
        user_text = clean_text(user_text)
        self.save_message(session_id, "user", user_text)

        if self.is_safety_risk(user_text):
            msg = ("I'm really concerned by what you wrote. If you're in immediate danger or thinking about harming yourself, "
                   "please contact local emergency services right away or call your local suicide prevention hotline (in the U.S., dial 988). "
                   "Are you safe right now?")
            self.save_message(session_id, "assistant", msg)
            return {"reply": msg, "mode": "safety", "debug": None, "ended": False}

        problem_flag = self.is_problem_statement(user_text)
        info = None
        if problem_flag:
            info = self.infer_emotions_from_text(user_text, top_k=top_k or self.DEFAULT_TOP_K)
            if info.get("error"):
                problem_flag = False

        messages = self._build_messages_with_context(session_id, user_text, problem_mode=problem_flag, info=info)
        reply = self.call_openai_chat_rest(messages)

        # remove possible leading name
        reply = re.sub(r'^\s*ron[:\-\s]*', '', reply, flags=re.I).strip()
        self.save_message(session_id, "assistant", reply)

        ended = self.is_user_satisfied(user_text)
        if ended:
            try:
                self.mark_conversation_ended(session_id)
            except Exception:
                pass

        return {
            "reply": reply,
            "mode": "problem" if problem_flag else "freechat",
            "debug": {"inferred": info.get("inferred") if info else None},
            "ended": ended
        }

    # ---------------- helpers ----------------
    def embeddings_exist(self) -> bool:
        try:
            return int(self.collection.count()) > 0
        except Exception:
            return False

    def collection_count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0
