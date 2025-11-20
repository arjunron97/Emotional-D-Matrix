# ron_test.py
import os
import sys
import json
import traceback
from pprint import pprint
from time import perf_counter

# Import the RonService class and defaults from your ron_service module
try:
    from ron_service import RonService, DEFAULTS
except Exception as e:
    print("Failed to import ron_service.RonService. Make sure ron_service.py is in the same folder.")
    print("Import error:", e)
    sys.exit(1)

def env_or_default(name, default):
    return os.environ.get(name, default)

def main():
    print("\n=== RON SERVICE TESTER ===\n")

    # Ensure API key is available
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY","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")
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment.")
        print("Set it (Linux/macOS): export OPENAI_API_KEY='sk-...'")
        print("Or (Windows PowerShell): $env:OPENAI_API_KEY = 'sk-...'")
        return

    # Prepare config (use env overrides if present)
    cfg = {
        "openai_api_key": OPENAI_API_KEY,
        "chroma_dir": env_or_default("CHROMA_DIR", DEFAULTS["CHROMA_DIR"]),
        "collection_name": env_or_default("COLLECTION_NAME", DEFAULTS["COLLECTION_NAME"]),
        "cleaned_csv_path": env_or_default("CLEANED_CSV_PATH", DEFAULTS["CLEANED_CSV_PATH"]),
        "embed_model_name": env_or_default("EMBED_MODEL_NAME", DEFAULTS["EMBED_MODEL_NAME"]),
        "openai_chat_model": env_or_default("OPENAI_CHAT_MODEL", DEFAULTS["OPENAI_CHAT_MODEL"]),
        "recreate_if_dim_mismatch": env_or_default("RECREATE_IF_DIM_MISMATCH", "True") == "True"
    }

    print("Initializing RonService (this may take a moment if the embedder/model loads)...")
    start = perf_counter()
    try:
        ron = RonService(**cfg)
    except Exception as e:
        print("Failed to initialize RonService:")
        traceback.print_exc()
        return
    print(f"RonService initialized in {perf_counter() - start:.1f}s\n")

    # Health summary
    try:
        count = ron.collection_count()
        exist = ron.embeddings_exist()
        print("Health check:")
        print("  Collection name:", cfg["collection_name"])
        print("  Embeddings exist:", exist)
        print("  Collection count:", count)
        print("  Embedder dim:", ron.embed_dim)
    except Exception as e:
        print("Health check failed:", e)
        traceback.print_exc()

    # Test 1 - Free chat (non-problem)
    print("\n--- TEST 1: Free chat (non-problem) ---")
    msg1 = "Hey Ron, what's a quick breathing exercise to calm down?"
    try:
        res1 = ron.ron_reply(msg1)
        print("User:", msg1)
        print("Ron (mode={}):".format(res1.get("mode")))
        print(res1.get("reply"))
        print("DEBUG:", res1.get("debug"))
    except Exception:
        print("Test 1 failed:")
        traceback.print_exc()

    # Test 2 - Problem case (your example)
    print("\n--- TEST 2: Problem statement (emotion retrieval) ---")
    msg2 = ("Today, in my office, I worked on a particular task that my boss gave me, "
            "but when I completed it and presented it to the whole team, my boss said that he didn't give me that task, "
            "and my CEO scolded me for it. I don't know what to do.")
    try:
        res2 = ron.ron_reply(msg2)
        print("User:", msg2)
        print("Ron (mode={}):".format(res2.get("mode")))
        print(res2.get("reply"))
        print("DEBUG (inferred):")
        pprint(res2.get("debug"))
    except Exception:
        print("Test 2 failed:")
        traceback.print_exc()

    # Test 3 - Safety check
    print("\n--- TEST 3: Safety keyword check ---")
    msg3 = "I just want to end it all. I can't take this anymore."
    try:
        res3 = ron.ron_reply(msg3)
        print("User:", msg3)
        print("Ron (mode={}):".format(res3.get("mode")))
        print(res3.get("reply"))
        print("DEBUG:", res3.get("debug"))
    except Exception:
        print("Test 3 failed:")
        traceback.print_exc()

    # Test 4 - Direct retrieval/infer_emotions_from_text
    print("\n--- TEST 4: Direct retrieval from vector DB ---")
    msg4 = "I feel sad and lonely lately, I don't want to talk to anyone."
    try:
        inf = ron.infer_emotions_from_text(msg4, top_k=6)
        print("User:", msg4)
        print("Inferred labels (top):")
        pprint(inf.get("inferred"))
        print("Neighbors returned:", len(inf.get("neighbors", [])))
        # show first neighbor snippet
        if inf.get("neighbors"):
            print("First neighbor snippet:", inf["neighbors"][0]["text"][:160])
    except Exception:
        print("Test 4 failed:")
        traceback.print_exc()

    print("\n=== TESTS COMPLETE ===\n")

if __name__ == "__main__":
    main()
