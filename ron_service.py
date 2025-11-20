# ron_service.py
import os
import time
import json
import re
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

import requests
import pandas as pd
import ftfy
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ---------------- CONFIG (edit as needed) ----------------
DEFAULTS = {
    "CHROMA_DIR": "./chroma_db",
    "COLLECTION_NAME": "goemotions_dataset",
    "CLEANED_CSV_PATH": "notebook/dataset/goemotions_cleaned.csv",
    "EMBED_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
    "EMBED_DIM_EXPECTED": 384,
    "OPENAI_CHAT_MODEL": "gpt-3.5-turbo",
    "RECREATE_IF_DIM_MISMATCH": True,
    "DEFAULT_TOP_K": 6,
    "REQUEST_TIMEOUT": 30.0,
    "MAX_RETRIES": 3,
    "RETRY_BACKOFF": 1.5,
    "OPENAI_API_BASE": "https://api.openai.com/v1",
}
# --------------------------------------------------------

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
    def __init__(self,
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
                 openai_api_base: str = DEFAULTS["OPENAI_API_BASE"]):
        # config + API key
        self.OPENAI_API_KEY = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            raise RuntimeError("OpenAI API key required: set OPENAI_API_KEY env var or pass into RonService().")
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

        # load embedder
        print("[RonService] Loading embedder:", self.EMBED_MODEL_NAME)
        self.embedder = SentenceTransformer(self.EMBED_MODEL_NAME)
        self.embed_dim = self.embedder.get_sentence_embedding_dimension()
        print("[RonService] Embedder dim:", self.embed_dim)
        if self.embed_dim != self.EMBED_DIM_EXPECTED:
            print(f"[RonService] Warning: embedder dim {self.embed_dim} != expected {self.EMBED_DIM_EXPECTED}; continuing with embedder dim.")

        # open chroma persistent client
        print("[RonService] Opening Chroma at:", self.CHROMA_DIR)
        self.client_db = PersistentClient(path=self.CHROMA_DIR)

        # list existing collections (Collection objects -> .name)
        try:
            existing = self.client_db.list_collections()
            self.existing_names = [c.name for c in existing]
        except Exception as e:
            print("[RonService] Warning listing collections:", e)
            self.existing_names = []

        # ensure collection ready
        self.collection = None
        self._ensure_collection_ready()

        # headers for REST OpenAI
        self.HEADERS = {
            "Authorization": f"Bearer {self.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        # heuristics
        self.PROBLEM_KEYWORDS = [
            "problem", "issue", "need help", "help me", "i can't", "i cannot", "i don't know",
            "i'm stuck", "i am stuck", "confused", "scared", "depressed", "can't handle",
            "anxious", "panic", "panic attack", "breakdown", "stress", "stressed", "overwhelmed",
            "scolded", "fired", "quit", "abuse", "harassed"
        ]
        self.SAFETY_KEYWORDS = [
            "suicide", "kill myself", "want to die", "end my life", "hurting myself", "commit suicide",
            "i will kill myself", "i am going to kill myself", "i want to die"
        ]

        print("[RonService] Ready.")

    # ---------- collection helpers ----------
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
        """
        Try to reliably create a fresh empty collection with the configured name.
        Handles cases where delete_collection might fail on some platforms.
        """
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
            # try create fresh; if UniqueConstraint appears, try get_or_create
            try:
                newcol = self.client_db.create_collection(name=name)
                return newcol
            except Exception:
                return self.client_db.get_or_create_collection(name=name)
        except Exception as e:
            # last resort: try create_collection directly
            try:
                return self.client_db.create_collection(name=name)
            except Exception as e2:
                print("[RonService] safe recreate failed:", e, e2)
                raise

    def _ensure_collection_ready(self):
        name = self.COLLECTION_NAME
        if name in self.existing_names:
            try:
                self.collection = self.client_db.get_collection(name)
                mismatch, expected, got = self._detect_collection_dim_mismatch(self.collection)
                if mismatch:
                    print(f"[RonService] Embedding-dim mismatch detected (expected={expected}, got={got}).")
                    if self.RECREATE_IF_DIM_MISMATCH:
                        print("[RonService] Recreating collection (clearing existing data).")
                        self.collection = self._safe_recreate_collection()
                    else:
                        raise RuntimeError("Embedding-dim mismatch and recreate disabled.")
                else:
                    print("[RonService] Using existing collection:", name)
            except Exception as e:
                print("[RonService] Error loading collection; attempting to create new. Error:", e)
                self.collection = self._safe_recreate_collection()
        else:
            print(f"[RonService] Collection '{name}' not present; creating.")
            self.collection = self._safe_recreate_collection()

        # if empty and CSV exists -> build
        try:
            cnt = 0
            try:
                cnt = self.collection.count()
            except Exception:
                cnt = 0
            print("[RonService] Collection count:", cnt)
            if int(cnt) == 0:
                p = Path(self.CLEANED_CSV_PATH)
                if p.exists():
                    print("[RonService] Collection empty and cleaned CSV found; building index.")
                    self._build_collection_from_csv(self.CLEANED_CSV_PATH, self.collection)
                else:
                    print("[RonService] Collection empty and cleaned CSV not found at:", self.CLEANED_CSV_PATH)
            else:
                print("[RonService] Collection already populated; skipping build.")
        except Exception as e:
            print("[RonService] Error during collection build-check:", e)
            traceback.print_exc()

    def _build_collection_from_csv(self, csv_path: str, col_obj, batch_size: int = 512):
        p = Path(csv_path)
        df = pd.read_csv(p, dtype=str, low_memory=False).fillna("")
        if "text" not in df.columns:
            raise RuntimeError("Cleaned CSV must contain 'text' column.")
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
        print("[RonService] Rows to index:", n)
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
        print("[RonService] CSV indexing complete.")

    # ---------- OpenAI REST wrapper ----------
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
                # retry rate-limits and server errors
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

    # ---------- retrieval + inference ----------
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

    # ---------- prompt building ----------
    SYSTEM_PROMPT = (
        "You are Ron — a calm, empathetic AI companion. You are not a medical professional. "
        "Listen, validate feelings, offer immediate coping suggestions and medium-term ideas. "
        "Keep replies warm, concise, and practical. If user expresses self-harm or imminent danger, "
        "provide emergency wording and encourage contacting local emergency services or a suicide hotline."
    )

    def _build_problem_messages(self, user_text: str, info: dict) -> List[Dict[str, str]]:
        inferred = info.get("inferred", [])
        inferred_str = ", ".join([f"{x['label']}({x['confidence']:.2f})" for x in inferred]) if inferred else "unknown"
        neighbors = info.get("neighbors", [])
        examples = "\n".join([f"- \"{n['text'][:140]}\" labels:{n['meta'].get('labels')} dist={n['distance']:.4f}" for n in neighbors[:3]]) or "none"
        user_block = (
            f"User: {user_text}\n\n"
            f"Inferred emotions: {inferred_str}\n"
            f"Top neighbor examples:\n{examples}\n\n"
            "Task: 1) Validate briefly. 2) Name likely emotion(s) and reason. 3) Give 2 short immediate coping actions. "
            "4) Give 2 medium-term suggestions. 5) When to seek professional help and emergency wording if needed. 6) End with a gentle follow-up question."
        )
        return [{"role":"system","content":self.SYSTEM_PROMPT},{"role":"user","content":user_block}]

    def _build_freechat_messages(self, user_text: str) -> List[Dict[str, str]]:
        return [{"role":"system","content":self.SYSTEM_PROMPT},{"role":"user","content":f"User: {user_text}\n\nTask: reply conversationally (max two short paragraphs). End with a follow-up question."}]

    # ---------- heuristics ----------
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

    # ---------- main API ----------
    def ron_reply(self, user_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        user_text = clean_text(user_text)
        if self.is_safety_risk(user_text):
            return {"reply": ("I’m really concerned by what you wrote. If you are in immediate danger or thinking about harming yourself, "
                              "please contact local emergency services right away or call your local suicide prevention hotline (in the U.S., dial 988). "
                              "I’m not a replacement for professional help. Are you safe right now?"), "mode":"safety", "debug": None}

        if self.is_problem_statement(user_text):
            info = self.infer_emotions_from_text(user_text, top_k=top_k or self.DEFAULT_TOP_K)
            if info.get("error"):
                messages = self._build_freechat_messages(user_text)
                out = self.call_openai_chat_rest(messages)
                return {"reply": out, "mode":"fallback", "debug": {"error": info.get("error")}}
            messages = self._build_problem_messages(user_text, info)
            out = self.call_openai_chat_rest(messages)
            return {"reply": out, "mode":"problem", "debug": {"inferred": info.get("inferred"), "neighbors": len(info.get("neighbors", []))}}
        else:
            messages = self._build_freechat_messages(user_text)
            out = self.call_openai_chat_rest(messages)
            return {"reply": out, "mode":"freechat", "debug": None}

    # ---------- small helpers for app health ----------
    def embeddings_exist(self) -> bool:
        try:
            c = self.collection.count()
            return int(c) > 0
        except Exception:
            return False

    def collection_count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0
