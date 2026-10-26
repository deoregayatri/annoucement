"""
test_llm_single_server.py
=========================
Standalone test that hits ONE specific Ollama LLM server directly.

Bypasses ALL of the production machinery:
  - No env / settings loading
  - No semaphore / rate limiter
  - No LLM_MAX_CONCURRENT cap
  - No shared httpx client
  - No token-bucket

Use this to:
  1. Verify a specific server is alive and returning responses.
  2. Diagnose timeout / retry failures by isolating one server.
  3. Compare response quality / speed across servers.

Usage:
  python test_llm_single_server.py

Edit the CONFIG section below to point at your target server.
"""

import json
import sys
import time
import httpx

# ---------------------------------------------------------------------------
# CONFIG — edit these values to target a specific server
# ---------------------------------------------------------------------------

# The Ollama server you want to test directly
LLM_SERVER_URL = "http://172.17.51.201:11434"   # <-- change to your server IP/port

# Model to use (must be pulled on that server)
MODEL = "qwen3:30b-a3b"                          # <-- change to your model name

# Test prompt
PROMPT = """You are a financial analyst. Answer in 2-3 sentences.

Question: What is the significance of EBITDA in evaluating company performance?"""

# Request parameters
NUM_CTX      = 4096    # context window (lower = faster for tests)
NUM_PREDICT  = 256     # max tokens to generate (keep small for quick tests)
TEMPERATURE  = 0.1
TIMEOUT_S    = 180     # seconds before giving up

# Number of test rounds (useful for spotting intermittent hangs)
ROUNDS = 1

# Disable qwen3 think-block (set False if using a non-qwen3 model)
DISABLE_THINK = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> CoT blocks from qwen3 output."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_ollama(prompt: str, round_num: int = 1) -> dict:
    """
    POST directly to the Ollama /api/generate endpoint on the configured server.
    Returns a dict with keys: success, response, elapsed_ms, error.
    """
    url = f"{LLM_SERVER_URL.rstrip('/')}/api/generate"

    payload: dict = {
        "model":       MODEL,
        "prompt":      prompt,
        "stream":      False,
        "num_ctx":     NUM_CTX,
        "num_predict": NUM_PREDICT,
        "temperature": TEMPERATURE,
    }

    if DISABLE_THINK and "qwen3" in MODEL.lower():
        payload["options"] = {"think": False}

    print(f"\n{'='*60}")
    print(f"Round {round_num} — POST {url}")
    print(f"Model       : {MODEL}")
    print(f"Timeout     : {TIMEOUT_S}s")
    print(f"think=False : {DISABLE_THINK and 'qwen3' in MODEL.lower()}")
    print(f"{'='*60}")
    print(f"Prompt preview: {prompt[:120].strip()}...")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    try:
        with httpx.Client(
            timeout=httpx.Timeout(
                connect=5.0,
                read=float(TIMEOUT_S),
                write=10.0,
                pool=5.0,
            )
        ) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        raw = resp.json().get("response", "")
        answer = _strip_think_tags(raw)

        return {
            "success":    True,
            "response":   answer,
            "raw":        raw,
            "elapsed_ms": elapsed_ms,
            "error":      None,
        }

    except httpx.TimeoutException as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "success":    False,
            "response":   "",
            "raw":        "",
            "elapsed_ms": elapsed_ms,
            "error":      f"TIMEOUT after {elapsed_ms/1000:.1f}s: {exc}",
        }
    except httpx.HTTPStatusError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "success":    False,
            "response":   "",
            "raw":        "",
            "elapsed_ms": elapsed_ms,
            "error":      f"HTTP {exc.response.status_code}: {exc}",
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "success":    False,
            "response":   "",
            "raw":        "",
            "elapsed_ms": elapsed_ms,
            "error":      f"{type(exc).__name__}: {exc}",
        }


def check_server_health() -> bool:
    """Quick GET /api/tags to verify the server is reachable before testing."""
    url = f"{LLM_SERVER_URL.rstrip('/')}/api/tags"
    print(f"[Health check] GET {url}")
    try:
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"[Health check] OK — available models: {models}")
        if MODEL not in models:
            print(f"\n  WARNING: '{MODEL}' is NOT in the model list above.")
            print(f"  Run:  ollama pull {MODEL}  on the server to fix this.\n")
        return True
    except Exception as exc:
        print(f"[Health check] FAILED — server unreachable: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LLM Single-Server Direct Test")
    print(f"  Server : {LLM_SERVER_URL}")
    print(f"  Model  : {MODEL}")
    print(f"  Rounds : {ROUNDS}")
    print("="*60)

    # 1. Health check
    if not check_server_health():
        print("\nAborting — server is not reachable.")
        sys.exit(1)

    # 2. Run test rounds
    all_results = []
    for i in range(1, ROUNDS + 1):
        result = call_ollama(PROMPT, round_num=i)
        all_results.append(result)

        if result["success"]:
            print(f"[Round {i}] SUCCESS  ({result['elapsed_ms']:.0f} ms)")
            print(f"\nAnswer:\n{result['response']}\n")
        else:
            print(f"[Round {i}] FAILED   ({result['elapsed_ms']:.0f} ms)")
            print(f"Error: {result['error']}\n")

        if i < ROUNDS:
            time.sleep(1)   # small gap between rounds

    # 3. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    successes = [r for r in all_results if r["success"]]
    failures  = [r for r in all_results if not r["success"]]
    avg_ms    = sum(r["elapsed_ms"] for r in successes) / len(successes) if successes else 0

    print(f"  Total rounds : {ROUNDS}")
    print(f"  Succeeded    : {len(successes)}")
    print(f"  Failed       : {len(failures)}")
    if successes:
        print(f"  Avg latency  : {avg_ms:.0f} ms  ({avg_ms/1000:.1f}s)")
        print(f"  Min latency  : {min(r['elapsed_ms'] for r in successes):.0f} ms")
        print(f"  Max latency  : {max(r['elapsed_ms'] for r in successes):.0f} ms")
    if failures:
        print(f"\nFailure details:")
        for r in failures:
            print(f"  - {r['error']}")

    print("="*60)
    sys.exit(0 if not failures else 1)
