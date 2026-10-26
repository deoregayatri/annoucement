"""
llm_client.py
=============
Single control point for ALL LLM (Ollama) calls in this system.

Provides:
  1. Shared httpx.AsyncClient - connection pool reused across all requests
     (eliminates the ~100-300ms overhead of a new TCP connection per call).

  2. asyncio.Semaphore(LLM_MAX_CONCURRENT=2) - hard concurrency cap.
     Derived from server capacity: 2 Ollama servers x 1 inflight req each.
     qwen3:30b-a3b on CPU is memory-bound; >1 inflight per server degrades all.

  3. Token-bucket rate limiter - absorbs burst storms after a quiet period.
     Prevents 20 queued requests from all firing simultaneously.

  4. Per-attempt timeout + exponential-backoff retry.
     Default: timeout=120s, retries=3, backoff=4s/8s/16s (capped 30s).

  5. Structured observability - every call logs:
     model, elapsed_ms, attempt number, semaphore_wait_ms.

  6. Graceful degradation - never raises; returns an error string on failure.

Usage:
  from agents.llm_client import call_llm_async, call_llm_sync

  # In async context (rag_graph.py nodes):
  answer = await call_llm_async(prompt)

  # In sync context (handlers.py, dictionary extraction):
  answer = call_llm_sync(prompt)
"""

import asyncio
import logging
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HTTP client (one instance per event loop)
# ---------------------------------------------------------------------------
_http_client: Optional[httpx.AsyncClient] = None
_client_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_http_client() -> httpx.AsyncClient:
    """Return (or create/recreate) the shared keep-alive HTTP client.

    Recreates the client if:
    - It was explicitly closed (is_closed=True)
    - The event loop changed (old client is bound to dead loop)
    """
    global _http_client, _client_loop

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    needs_recreate = False

    if _http_client is None or _http_client.is_closed:
        needs_recreate = True
    elif _client_loop is not None and current_loop is not _client_loop:
        # Event loop changed - old client is bound to dead loop
        needs_recreate = True

    if needs_recreate:
        from config import settings

        timeout_s = float(getattr(settings, "llm_request_timeout_s", 120))
        _http_client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=timeout_s,
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=4,
                keepalive_expiry=60.0,
            ),
        )
        _client_loop = current_loop
        logger.info(
            f"[LLMClient] AsyncClient created: base_url={settings.ollama_base_url} "
            f"timeout={timeout_s}s"
        )
    return _http_client


# ---------------------------------------------------------------------------
# Concurrency semaphore - one per asyncio event loop
# ---------------------------------------------------------------------------
_semaphore: Optional[asyncio.Semaphore] = None
_current_concurrency: int = 2


def _get_semaphore() -> asyncio.Semaphore:
    """
    Return (or create) the global LLM concurrency semaphore.
    Capacity defaults to LLM_MAX_CONCURRENT in settings, but can be scaled
    dynamically at runtime based on available load balancer infrastructure.
    """
    global _semaphore, _current_concurrency
    if _semaphore is None:
        from config import settings

        _current_concurrency = int(getattr(settings, "llm_max_concurrent", 2))
        _semaphore = asyncio.Semaphore(_current_concurrency)
        logger.info(
            f"[LLMClient] Semaphore created: max_concurrent={_current_concurrency}"
        )
    return _semaphore


def update_llm_concurrency(new_limit: int) -> None:
    """
    Dynamically scale the in-flight concurrency limit at runtime.
    Replaces the global semaphore. In-flight tasks bound to the old semaphore
    will finish and release natively. New tasks will queue on the new capacity.
    """
    global _semaphore, _current_concurrency
    if new_limit < 1:
        logger.warning(f"[LLMClient] Invalid concurrency limit {new_limit}, ignoring.")
        return

    _current_concurrency = new_limit
    _semaphore = asyncio.Semaphore(new_limit)
    logger.info(
        f"[LLMClient] Concurrency dynamically scaled to: max_concurrent={new_limit}"
    )


# ---------------------------------------------------------------------------
# Token-bucket rate limiter - prevents burst storms
# ---------------------------------------------------------------------------
class _TokenBucket:
    """
    Limits burst: allows up to `capacity` back-to-back calls,
    then enforces `rate` new tokens per second.

    rate=0.5 ? one new token every 2 seconds, matching ~20-60s LLM generation.
    capacity=4 ? burst headroom = 2 x server count.
    """

    def __init__(self, rate: float = 0.5, capacity: int = 4):
        self._rate = rate
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._ts = time.monotonic()
        self._alock = None  # async lock, created on first acquire

    async def acquire(self) -> None:
        if self._alock is None:
            self._alock = asyncio.Lock()
        async with self._alock:
            now = time.monotonic()
            earned = (now - self._ts) * self._rate
            self._tokens = min(self._capacity, self._tokens + earned)
            self._ts = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
        # Insufficient tokens - wait outside the lock
        wait_s = (1.0 - self._tokens) / self._rate
        logger.debug(f"[LLMClient] Rate-limit wait {wait_s:.1f}s")
        await asyncio.sleep(wait_s)
        if self._alock:
            async with self._alock:
                self._tokens = 0.0


_rate_limiter = _TokenBucket(rate=0.5, capacity=4)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def call_llm_async(
    prompt: str,
    model: str = None,
    json_format: bool = False,
    temperature: float = None,
) -> str:
    """
    Internal async LLM execution orchestrator for NEAPS/RAG pipelines.
    Enforces concurrency and rate limits logic.

    Flow per request:
      rate-limit ? semaphore_wait ? HTTP POST ? retry-on-failure

    Observability logged per call:
      - semaphore_wait_ms  (time spent waiting for a slot)
      - llm_elapsed_ms     (time from POST send to response received)
      - attempt number
      - total_ms           (end-to-end including retries)

    Returns:
      Response string with <think> tags stripped.
      On permanent failure: "[LLM Error: ...]" - never raises.
    """
    from config import settings
    from agents.llm_utils import _strip_think_tags

    model = model or settings.ollama_model_answer
    temp = (
        temperature if temperature is not None else settings.ollama_temperature_answer
    )
    timeout_s = float(getattr(settings, "llm_request_timeout_s", 120))
    max_retries = int(getattr(settings, "llm_max_retries", 3))
    base_wait = float(getattr(settings, "llm_retry_base_s", 4))

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "num_ctx": settings.ollama_num_ctx,
        "num_predict": int(getattr(settings, "ollama_num_predict", 1024)),
        "temperature": temp,
        **({"format": "json"} if json_format else {}),
    }

    # Disable qwen3's extended <think>…</think> chain-of-thought.
    # This is the single biggest latency driver: on qwen3:30b-a3b it can add
    # 60–120s of token generation before the actual answer even starts.
    # think=False tells Ollama to skip the CoT block entirely.
    if "qwen3" in model.lower():
        payload["options"] = payload.get("options", {})
        payload["options"]["think"] = False
        logger.debug(f"[LLMClient] qwen3 detected — think=False injected")

    call_start = time.perf_counter()

    # Step 1: token-bucket rate limit (non-blocking burst control)
    await _rate_limiter.acquire()

    # Step 2: semaphore (hard concurrency cap = number of Ollama servers)
    sem = _get_semaphore()
    client = _get_http_client()

    for attempt in range(1, max_retries + 1):
        sem_wait_start = time.perf_counter()
        try:
            async with sem:
                sem_wait_ms = (time.perf_counter() - sem_wait_start) * 1000
                llm_start = time.perf_counter()

                resp = await client.post(
                    "/api/generate", json=payload, timeout=timeout_s
                )
                resp.raise_for_status()

                llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
                total_ms = (time.perf_counter() - call_start) * 1000
                raw = resp.json().get("response", "")

                logger.info(
                    f"[LLMClient] OK | model={model} attempt={attempt} "
                    f"sem_wait={sem_wait_ms:.0f}ms "
                    f"llm_elapsed={llm_elapsed_ms:.0f}ms "
                    f"total={total_ms:.0f}ms "
                    f"response_len={len(raw)}"
                )
                return _strip_think_tags(raw)

        except httpx.TimeoutException as exc:
            logger.warning(
                f"[LLMClient] TIMEOUT | model={model} attempt={attempt}/{max_retries} "
                f"after {timeout_s}s: {exc}"
            )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                f"[LLMClient] HTTP_ERROR | model={model} attempt={attempt}/{max_retries} "
                f"status={exc.response.status_code}: {exc}"
            )
        except Exception as exc:
            logger.warning(
                f"[LLMClient] ERROR | model={model} attempt={attempt}/{max_retries}: {exc}"
            )
            # If event loop closed, recreate client for next retry
            if "Event loop is closed" in str(exc):
                global _http_client, _client_loop
                _http_client = None
                _client_loop = None
                client = _get_http_client()

        if attempt < max_retries:
            wait = min(base_wait * (2 ** (attempt - 1)), 30)  # 4s, 8s, 16s ? cap 30s
            logger.info(
                f"[LLMClient] Retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(wait)

    total_ms = (time.perf_counter() - call_start) * 1000
    logger.error(
        f"[LLMClient] FAILED | model={model} all {max_retries} retries exhausted "
        f"total={total_ms:.0f}ms"
    )
    return "{}" if json_format else "[LLM Error: all retries exhausted]"


# ---------------------------------------------------------------------------
# Dedicated background event loop — fixes semaphore/event-loop deadlock
# ---------------------------------------------------------------------------
# ROOT CAUSE (production bug):
#   asyncio.run() creates a NEW event loop on every call_llm_sync() invocation.
#   The global _semaphore was created on a PREVIOUS (now dead) event loop, so
#   it appears permanently locked from the new loop's perspective.
#   Result: 10 retries × 30 s = 300 s deadlock, then [LLM Error: all retries exhausted].
#
# FIX:
#   Run ONE persistent background event loop in a daemon thread.
#   All asyncio objects (_semaphore, _http_client) live inside that single loop
#   and are therefore always alive and reachable.
# ---------------------------------------------------------------------------

_sync_loop: Optional[asyncio.AbstractEventLoop] = None
_sync_loop_thread: Optional[threading.Thread] = None
_sync_loop_lock = threading.Lock()


def _get_sync_loop() -> asyncio.AbstractEventLoop:
    """
    Return (or lazily create) the single long-lived background event loop.
    The loop is started in a daemon thread and NEVER closed for the lifetime
    of the process, so all asyncio objects created inside it remain valid.
    """
    global _sync_loop, _sync_loop_thread
    with _sync_loop_lock:
        if _sync_loop is None or not _sync_loop.is_running():
            _sync_loop = asyncio.new_event_loop()
            _sync_loop_thread = threading.Thread(
                target=_sync_loop.run_forever,
                daemon=True,
                name="llm-sync-loop",
            )
            _sync_loop_thread.start()
            logger.info(
                "[LLMClient] Dedicated sync event loop started "
                f"(thread={_sync_loop_thread.name})"
            )
    return _sync_loop


# Sync wrapper (for handlers.py, dictionary extraction, and other sync callers)


def call_llm_sync(
    prompt: str,
    model: str = None,
    json_format: bool = False,
    temperature: float = None,
) -> str:
    """
    Synchronous wrapper — submits the coroutine to the persistent background loop.

    Replaces asyncio.run() which was creating a throwaway event loop on every
    call, causing the global semaphore (bound to a previous loop) to appear
    permanently locked, resulting in 300-second deadlocks in production.

    Safe for:
    - ProcessPoolExecutor workers (each gets its own persistent loop)
    - ThreadPoolExecutor handlers (shared loop, semaphore still respected)
    - Sync graph nodes via graph.invoke()
    """
    loop = _get_sync_loop()
    future = asyncio.run_coroutine_threadsafe(
        call_llm_async(prompt, model, json_format, temperature),
        loop,
    )
    timeout_s = float(getattr(_get_settings_safe(), "llm_request_timeout_s", 120))
    # Allow 3× the per-attempt timeout to cover all retries + backoff
    total_wait = timeout_s * 3 + 60
    try:
        return future.result(timeout=total_wait)
    except Exception as exc:
        logger.error(f"[LLMClient] call_llm_sync failed: {exc}")
        return "{}" if json_format else f"[LLM Error: {exc}]"


def _get_settings_safe():
    try:
        from config import settings

        return settings
    except Exception:
        return type("S", (), {"llm_request_timeout_s": 120})()


# ---------------------------------------------------------------------------
# Shutdown hook - close the shared HTTP client gracefully
# ---------------------------------------------------------------------------


async def close_llm_client() -> None:
    """Call from FastAPI lifespan shutdown to release connections."""
    global _http_client, _client_loop
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        logger.info("[LLMClient] AsyncClient closed.")
        _http_client = None
        _client_loop = None


# ===========================================================================
# VLM (Vision Language Model) - SEPARATE from text LLM
# ===========================================================================
#
# Concurrency model:
#   VLM_CONCURRENCY = VLM_SERVER_COUNT x VLM_MAX_CONCURRENCY_PER_SERVER
#   _vlm_semaphore  = asyncio.Semaphore(VLM_CONCURRENCY)
#
# This semaphore is completely independent of _semaphore (text LLM).
# VLM calls hit the same Ollama base_url but use a multimodal payload with
# the 'images' key. Concurrency is calibrated separately because VLM
# inference (image encode + generate) is slower than text-only calls.
# ===========================================================================

_vlm_semaphore: Optional[asyncio.Semaphore] = None
_vlm_http_client: Optional[httpx.AsyncClient] = None


def _get_vlm_semaphore() -> asyncio.Semaphore:
    """Return (or create) the VLM-specific concurrency semaphore."""
    global _vlm_semaphore
    if _vlm_semaphore is None:
        from config import settings

        concurrency = int(getattr(settings, "vlm_server_count", 1)) * int(
            getattr(settings, "vlm_max_concurrency_per_server", 1)
        )
        concurrency = max(1, concurrency)
        _vlm_semaphore = asyncio.Semaphore(concurrency)
        logger.info(f"[VLMClient] Semaphore created: max_concurrent={concurrency}")
    return _vlm_semaphore


def _get_vlm_http_client() -> httpx.AsyncClient:
    """Return (or create) a dedicated HTTP client for VLM calls."""
    global _vlm_http_client
    if _vlm_http_client is None or _vlm_http_client.is_closed:
        from config import settings

        timeout_s = float(getattr(settings, "vlm_request_timeout_s", 120))
        _vlm_http_client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=timeout_s,
                write=15.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=6,
                max_keepalive_connections=3,
                keepalive_expiry=60.0,
            ),
        )
        logger.info(
            f"[VLMClient] AsyncClient created: base_url={settings.ollama_base_url} "
            f"timeout={timeout_s}s"
        )
    return _vlm_http_client


async def call_vlm_async(
    images: list,  # list of base64-encoded image strings
    prompt: str = "Describe this image in detail.",
    model: str = None,
    metadata: dict = None,  # for logging context only
) -> str:
    """
    Send image(s) to the Ollama VLM endpoint with controlled concurrency.

    Uses the VLM-specific semaphore (_vlm_semaphore) - NOT the text LLM semaphore.
    Images are base64-encoded PNG strings; up to `vlm_images_per_request`
    are included per request.

    Flow per request:
      semaphore_wait ? HTTP POST /api/generate (multimodal) ? retry-on-failure

    Returns:
      VLM response string, or "[VLM Error: ...]" on permanent failure.
      Never raises.
    """
    from config import settings
    from agents.llm_utils import _strip_think_tags

    model = model or getattr(settings, "ollama_model_vision", "minicpm-v:latest")
    timeout_s = float(getattr(settings, "vlm_request_timeout_s", 120))
    max_retries = int(getattr(settings, "vlm_max_retries", 2))
    base_wait = 4.0

    if not images:
        return "[VLM Error: no images supplied]"

    # Safety: honour vlm_images_per_request cap
    max_per_req = max(1, min(3, int(getattr(settings, "vlm_images_per_request", 1))))
    images_to_send = images[:max_per_req]

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": images_to_send,
    }

    sem = _get_vlm_semaphore()
    client = _get_vlm_http_client()
    call_start = time.perf_counter()

    doc_label = (metadata or {}).get("ann_seq", "?")

    for attempt in range(1, max_retries + 1):
        sem_wait_start = time.perf_counter()
        try:
            async with sem:
                sem_wait_ms = (time.perf_counter() - sem_wait_start) * 1000
                vlm_start = time.perf_counter()

                resp = await client.post(
                    "/api/generate",
                    json=payload,
                    timeout=timeout_s,
                )
                resp.raise_for_status()

                vlm_elapsed_ms = (time.perf_counter() - vlm_start) * 1000
                total_ms = (time.perf_counter() - call_start) * 1000
                raw = resp.json().get("response", "")

                logger.info(
                    f"[VLMClient] OK | model={model} ann_seq={doc_label} "
                    f"images={len(images_to_send)} attempt={attempt} "
                    f"sem_wait={sem_wait_ms:.0f}ms "
                    f"vlm_elapsed={vlm_elapsed_ms:.0f}ms "
                    f"total={total_ms:.0f}ms "
                    f"response_len={len(raw)}"
                )
                return _strip_think_tags(raw)

        except httpx.TimeoutException as exc:
            logger.warning(
                f"[VLMClient] TIMEOUT | model={model} attempt={attempt}/{max_retries} "
                f"after {timeout_s}s: {exc}"
            )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                f"[VLMClient] HTTP_ERROR | model={model} attempt={attempt}/{max_retries} "
                f"status={exc.response.status_code}: {exc}"
            )
        except Exception as exc:
            logger.warning(
                f"[VLMClient] ERROR | model={model} attempt={attempt}/{max_retries}: {exc}"
            )

        if attempt < max_retries:
            wait = min(base_wait * (2 ** (attempt - 1)), 30)
            logger.info(
                f"[VLMClient] Retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(wait)

    total_ms = (time.perf_counter() - call_start) * 1000
    logger.error(
        f"[VLMClient] FAILED | model={model} ann_seq={doc_label} "
        f"all {max_retries} retries exhausted total={total_ms:.0f}ms"
    )
    return "[VLM Error: all retries exhausted]"


def call_vlm_sync(
    images: list,
    prompt: str = "Describe this image in detail.",
    model: str = None,
    metadata: dict = None,
) -> str:
    """
    Synchronous wrapper around call_vlm_async.
    Safe to call from ProcessPoolExecutor / ThreadPoolExecutor workers.

    Mirrors the pattern of call_llm_sync so it works correctly whether
    called inside or outside a running asyncio event loop.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    coro = call_vlm_async(images, prompt, model, metadata)

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            max_wait = (
                float(getattr(_get_settings_safe(), "vlm_request_timeout_s", 120)) * 3
                + 30
            )
            return future.result(timeout=max_wait)
    else:
        return asyncio.run(coro)


async def close_vlm_client() -> None:
    """Close the VLM HTTP client on shutdown."""
    global _vlm_http_client
    if _vlm_http_client and not _vlm_http_client.is_closed:
        await _vlm_http_client.aclose()
        logger.info("[VLMClient] AsyncClient closed.")
        _vlm_http_client = None
