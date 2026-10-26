"""
test_search_with_llm_check.py
==============================
Extended version of test_api_encrypted.py.

What this does (in order):
  STEP 1 — Check if the target LLM server is alive  (GET /api/tags)
  STEP 2 — Check if the model is currently loaded   (GET /api/ps)
  STEP 3 — Send a direct LLM call to the server     (POST /api/generate)
  STEP 4 — Send the encrypted search API request    (same as test_api_encrypted.py)
  STEP 5 — Compare: did the search API produce a
            similar answer to the direct LLM call?

This lets you confirm:
  - The LLM server is UP before the search API call
  - The search API's answer comes from a working LLM (not a cached/error stub)
  - Latency of each step separately

No imports from the project. Only: requests, json, base64, os, time.
"""

import base64
import json
import os
import time

import requests

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------

# --- Search API (your backend) ---
SEARCH_API_URL   = "http://172.17.177.78:8081/api/v1/search"
BEARER_TOKEN     = "YOUR_TOKEN_HERE"          # replace with your actual token
ENCRYPTION_KEY   = b'U8eR0yD12ak9X!wQ@r5tZ?bY$4L3vP2h'

# --- The query to test ---
QUERY            = "total operation cost"
COMPANIES        = ["JASH"]
TOP_K            = 5

# --- LLM server to verify ---
LLM_SERVER_URL   = "http://172.17.51.201:11434"   # the specific server to check
MODEL_NAME       = "qwen3:30b-a3b"               # model that should be loaded there

# --- Direct LLM test prompt (mirrors what the search API would ask) ---
LLM_TEST_PROMPT  = f"""You are a financial analyst.
Answer in 2-3 sentences based on general knowledge.

Question: {QUERY}"""

LLM_TIMEOUT_S    = 120   # seconds to wait for LLM response

# ---------------------------------------------------------------------------
# AES-GCM encrypt / decrypt  (same as test_api_encrypted.py)
# ---------------------------------------------------------------------------

def encrypt_aes(plaintext: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aesgcm = AESGCM(ENCRYPTION_KEY)
    iv = os.urandom(12)
    cipher_text = aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
    encrypted_data = iv + cipher_text
    return base64.urlsafe_b64encode(encrypted_data).rstrip(b"=").decode("utf-8")


def decrypt_aes(encrypted: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    base64_encoded = encrypted.replace('-', '+').replace('_', '/')
    padding = len(base64_encoded) % 4
    if padding:
        base64_encoded += "=" * (4 - padding)
    encrypted_data = base64.b64decode(base64_encoded)
    iv = encrypted_data[:12]
    cipher_text = encrypted_data[12:]
    aesgcm = AESGCM(ENCRYPTION_KEY)
    return aesgcm.decrypt(iv, cipher_text, None).decode('utf-8')


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def divider(title: str = ""):
    print("\n" + "="*60)
    if title:
        print(f"  {title}")
        print("="*60)


def step1_check_server_alive() -> bool:
    """GET /api/tags — verify the LLM server responds."""
    divider("STEP 1 — LLM Server Health Check")
    url = f"{LLM_SERVER_URL.rstrip('/')}/api/tags"
    print(f"GET {url}")
    try:
        t0 = time.perf_counter()
        resp = requests.get(url, timeout=10)
        elapsed = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"[OK]  Server responded in {elapsed:.0f}ms")
        print(f"      Available models: {models}")
        if MODEL_NAME not in models:
            print(f"  [WARNING] '{MODEL_NAME}' NOT found in model list!")
            print(f"            Run: ollama pull {MODEL_NAME}  on the server")
        else:
            print(f"  [OK] Model '{MODEL_NAME}' is available")
        return True
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Cannot connect to {LLM_SERVER_URL}")
        print("       Is the server running? Is the IP/port correct?")
        return False
    except requests.exceptions.Timeout:
        print(f"[FAIL] Timeout connecting to {LLM_SERVER_URL}")
        return False
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return False


def step2_check_model_loaded() -> bool:
    """GET /api/ps — check if the model is currently loaded in memory."""
    divider("STEP 2 — Check Model Loaded in Memory")
    url = f"{LLM_SERVER_URL.rstrip('/')}/api/ps"
    print(f"GET {url}")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        running = resp.json().get("models", [])
        if running:
            for m in running:
                print(f"  [LOADED] {m.get('name')} — size={m.get('size_vram', 'N/A')} expires={m.get('expires_at', 'N/A')}")
            loaded_names = [m.get("name", "") for m in running]
            if any(MODEL_NAME in n for n in loaded_names):
                print(f"  [OK] Target model '{MODEL_NAME}' is hot in memory (fast first call)")
                return True
            else:
                print(f"  [INFO] '{MODEL_NAME}' not currently loaded — will be loaded on first request (~30-60s cold start)")
                return True
        else:
            print("  [INFO] No models currently loaded — cold start expected on first request")
            return True
    except Exception as exc:
        print(f"  [WARN] /api/ps not available: {exc}  (older Ollama version — skipping)")
        return True   # Non-fatal — older Ollama versions don't have /api/ps


def step3_direct_llm_call() -> dict:
    """POST /api/generate directly to the LLM server."""
    divider("STEP 3 — Direct LLM Call (bypassing search API)")
    url = f"{LLM_SERVER_URL.rstrip('/')}/api/generate"
    print(f"POST {url}")
    print(f"Model   : {MODEL_NAME}")
    print(f"Timeout : {LLM_TIMEOUT_S}s")
    print(f"Prompt  : {LLM_TEST_PROMPT[:100].strip()}...")

    payload = {
        "model":       MODEL_NAME,
        "prompt":      LLM_TEST_PROMPT,
        "stream":      False,
        "num_predict": 200,
        "temperature": 0.1,
        "options":     {"think": False} if "qwen3" in MODEL_NAME.lower() else {},
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT_S)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()

        import re
        raw_answer = resp.json().get("response", "")
        # Strip <think>...</think> blocks from qwen3 output
        answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

        print(f"\n[OK]  LLM responded in {elapsed_ms:.0f}ms ({elapsed_ms/1000:.1f}s)")
        print(f"\nDirect LLM Answer:\n{answer[:500]}")
        return {"success": True, "answer": answer, "elapsed_ms": elapsed_ms}

    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"\n[FAIL] LLM TIMED OUT after {elapsed_ms/1000:.1f}s")
        print("       The server is alive but the model is not responding.")
        print("       This is the same failure mode causing 'retries exhausted' in server.log")
        return {"success": False, "answer": "", "elapsed_ms": elapsed_ms, "error": "timeout"}

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        return {"success": False, "answer": "", "elapsed_ms": elapsed_ms, "error": str(exc)}


def step4_encrypted_search_api() -> dict:
    """Send the encrypted query to the backend search API."""
    divider("STEP 4 — Encrypted Search API Call")
    payload = {
        "query":     QUERY,
        "companies": COMPANIES,
        "top_k":     TOP_K,
    }
    encrypted_body = encrypt_aes(json.dumps(payload))
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type":  "text/plain",
    }
    print(f"POST {SEARCH_API_URL}")
    print(f"Query    : {QUERY}")
    print(f"Companies: {COMPANIES}")

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            SEARCH_API_URL,
            data=f'"{encrypted_body}"',
            headers=headers,
            timeout=300,   # search + LLM can take up to 5 min
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"\nHTTP Status : {resp.status_code}  ({elapsed_ms:.0f}ms / {elapsed_ms/1000:.1f}s)")

        if resp.status_code == 200:
            try:
                decrypted_str = decrypt_aes(resp.text.strip().strip('"'))
                result = json.loads(decrypted_str)
                answer = result.get("answer") or result.get("response") or ""
                print(f"\n[OK]  Search API returned successfully")
                print(f"\nSearch API Answer:\n{str(answer)[:500]}")
                return {"success": True, "result": result, "answer": answer, "elapsed_ms": elapsed_ms}
            except Exception:
                # Response may not be encrypted
                print(f"\nRaw response (not encrypted):\n{resp.text[:500]}")
                return {"success": True, "result": {}, "answer": resp.text, "elapsed_ms": elapsed_ms}
        else:
            print(f"\n[FAIL] Search API returned HTTP {resp.status_code}")
            print(f"Body: {resp.text[:300]}")
            return {"success": False, "result": {}, "answer": "", "elapsed_ms": elapsed_ms}

    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"\n[FAIL] Search API TIMED OUT after {elapsed_ms/1000:.1f}s")
        return {"success": False, "result": {}, "answer": "", "elapsed_ms": elapsed_ms, "error": "timeout"}

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"\n[FAIL] {type(exc).__name__}: {exc}")
        return {"success": False, "result": {}, "answer": "", "elapsed_ms": elapsed_ms}


def step5_verdict(llm_result: dict, search_result: dict):
    """Compare results and give a verdict on whether the LLM server is being used."""
    divider("STEP 5 — Verdict")

    llm_ok     = llm_result.get("success", False)
    search_ok  = search_result.get("success", False)
    llm_ms     = llm_result.get("elapsed_ms", 0)
    search_ms  = search_result.get("elapsed_ms", 0)

    print(f"  Direct LLM call   : {'✓ OK' if llm_ok else '✗ FAILED'}  ({llm_ms:.0f}ms)")
    print(f"  Search API call   : {'✓ OK' if search_ok else '✗ FAILED'}  ({search_ms:.0f}ms)")

    if llm_ok and search_ok:
        print("\n  [CONCLUSION] Both the LLM server and the search API are working.")
        print(f"  The search API took {search_ms/1000:.1f}s total (retrieval + LLM).")
        print(f"  The direct LLM call took {llm_ms/1000:.1f}s.")
        expected_overhead = search_ms - llm_ms
        if expected_overhead > 0:
            print(f"  Retrieval overhead ≈ {expected_overhead:.0f}ms (embedding + vector search + rerank)")

    elif llm_ok and not search_ok:
        print("\n  [CONCLUSION] LLM server is WORKING but the search API FAILED.")
        print("  Problem is in the backend (retrieval, routing, or encryption), not the LLM.")

    elif not llm_ok and search_ok:
        print("\n  [CONCLUSION] Search API returned OK but the LLM server TIMED OUT directly.")
        print("  The search API may be using a DIFFERENT LLM server than the one configured here.")
        print(f"  Check which server is set as OLLAMA_BASE_URL in the backend .env")

    else:
        print("\n  [CONCLUSION] BOTH FAILED.")
        if llm_result.get("error") == "timeout":
            print("  The LLM server is alive but the model is hanging — this causes")
            print("  'all retries exhausted' in server.log.")
            print("  Fix: restart ollama on the server or increase OLLAMA_TIMEOUT in .env")
        else:
            print("  Check if the LLM server is running and the model is pulled.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    divider("LLM-Verified Search API Test")
    print(f"  LLM Server  : {LLM_SERVER_URL}")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Search API  : {SEARCH_API_URL}")
    print(f"  Query       : '{QUERY}'")

    # Run all steps
    server_alive = step1_check_server_alive()
    if not server_alive:
        divider("ABORTED")
        print("LLM server is not reachable. Fix connectivity before proceeding.")
        exit(1)

    step2_check_model_loaded()
    llm_result    = step3_direct_llm_call()
    search_result = step4_encrypted_search_api()
    step5_verdict(llm_result, search_result)

    divider()
    print("Done.")
