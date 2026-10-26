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
# Thread-safe & Event-loop-agnostic Semaphore
# ---------------------------------------------------------------------------
class ThreadSafeSemaphore:
    """
    Concurrency semaphore that is completely decoupled from any single asyncio event loop.
    Works seamlessly with both `with` (sync) and `async with` (async) across any thread,
    worker process, or event loop without 'bound to a different event loop' errors.
    """

    def __init__(self, value: int = 4):
        self._value = max(1, value)
        self._sem = threading.BoundedSemaphore(self._value)

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        if timeout is not None:
            return self._sem.acquire(blocking, timeout)
        return self._sem.acquire(blocking)

    def release(self) -> None:
        try:
            self._sem.release()
        except ValueError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    async def __aenter__(self):
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.acquire)
        except RuntimeError:
            self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


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
                max_connections=16,
                max_keepalive_connections=8,
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
# Concurrency semaphore - thread-safe across all threads and loops
# ---------------------------------------------------------------------------
_semaphore: Optional[ThreadSafeSemaphore] = None
_current_concurrency: int = 4


def _get_semaphore() -> ThreadSafeSemaphore:
    """
    Return (or create) the global LLM concurrency semaphore.
    Capacity defaults to LLM_MAX_CONCURRENT in settings (default 4 for 4 Ollama servers).
    """
    global _semaphore, _current_concurrency
    if _semaphore is None:
        from config import settings

        _current_concurrency = int(getattr(settings, "llm_max_concurrent", 4))
        _semaphore = ThreadSafeSemaphore(_current_concurrency)
        logger.info(
            f"[LLMClient] ThreadSafeSemaphore created: max_concurrent={_current_concurrency}"
        )
    return _semaphore


def update_llm_concurrency(new_limit: int) -> None:
    """
    Dynamically scale the in-flight concurrency limit at runtime.
    """
    global _semaphore, _current_concurrency
    if new_limit < 1:
        logger.warning(f"[LLMClient] Invalid concurrency limit {new_limit}, ignoring.")
        return

    _current_concurrency = new_limit
    _semaphore = ThreadSafeSemaphore(new_limit)
    logger.info(
        f"[LLMClient] Concurrency dynamically scaled to: max_concurrent={new_limit}"
    )


# ---------------------------------------------------------------------------
# Token-bucket rate limiter - prevents burst storms (thread-safe)
# ---------------------------------------------------------------------------
class _TokenBucket:
    """
    Limits burst: allows up to `capacity` back-to-back calls,
    then enforces `rate` new tokens per second.
    """

    def __init__(self, rate: float = 0.5, capacity: int = 4):
        self._rate = rate
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._ts = time.monotonic()
        self._lock = threading.Lock()

    async def acquire(self) -> None:
        wait_s = 0.0
        with self._lock:
            now = time.monotonic()
            earned = (now - self._ts) * self._rate
            self._tokens = min(self._capacity, self._tokens + earned)
            self._ts = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait_s = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0

        if wait_s > 0:
            logger.debug(f"[LLMClient] Rate-limit wait {wait_s:.1f}s")
            await asyncio.sleep(wait_s)


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
      rate-limit -> semaphore_wait -> HTTP POST -> retry-on-failure

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
    # Cap retries to 2 (max 2 attempts) to prevent long hangs that violate Akamai 5-min timeout
    max_retries = min(int(getattr(settings, "llm_max_retries", 2)), 2)
    base_wait = float(getattr(settings, "llm_retry_base_s", 2))

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
    # This is the single biggest latency driver: on qwen3:30b-a3b on CPU it can add
    # 60–180s of token generation before the actual answer even starts.
    # think=False tells Ollama to skip the CoT block entirely.
    if "qwen3" in (model or "").lower():
        payload["options"] = payload.get("options", {})
        payload["options"]["think"] = False
        logger.debug("[LLMClient] qwen3 detected — think=False injected into options")

    call_start = time.perf_counter()

    # Step 1: token-bucket rate limit (non-blocking burst control)
    await _rate_limiter.acquire()

    # Step 2: semaphore (hard concurrency cap = number of Ollama servers)
    sem = _get_semaphore()
    client = _get_http_client()

    # One TimingLogger per LLM call — carries model+prompt fingerprint as document_id
    from utils.timing_log import TimingLogger as _TL
    _tl = _TL(
        document_id=f"model={model}",
        worker_id=f"PID-{__import__('os').getpid()}",
    )

    for attempt in range(1, max_retries + 1):
        sem_wait_start = time.perf_counter()
        _tl.emit("LLM_QUEUE", attempt=attempt)
        try:
            async with sem:
                sem_wait_ms = (time.perf_counter() - sem_wait_start) * 1000
                llm_start = time.perf_counter()
                _tl.emit("LLM_START", attempt=attempt, sem_wait_ms=f"{sem_wait_ms:.0f}")

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
                _tl.emit("LLM_END", attempt=attempt,
                         llm_ms=f"{llm_elapsed_ms:.0f}", tokens=len(raw.split()))
                return _strip_think_tags(raw)

        except httpx.TimeoutException as exc:
            logger.warning(
                f"[LLMClient] TIMEOUT | model={model} attempt={attempt}/{max_retries} "
                f"after {timeout_s}s: {exc}"
            )
            _tl.emit_error("LLM_ERROR", error=f"TIMEOUT after {timeout_s}s", attempt=attempt)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                f"[LLMClient] HTTP_ERROR | model={model} attempt={attempt}/{max_retries} "
                f"status={exc.response.status_code}: {exc}"
            )
            _tl.emit_error("LLM_ERROR", error=f"HTTP {exc.response.status_code}", attempt=attempt)
        except Exception as exc:
            logger.warning(
                f"[LLMClient] ERROR | model={model} attempt={attempt}/{max_retries}: {exc}"
            )
            _tl.emit_error("LLM_ERROR", error=str(exc), attempt=attempt)
            # If event loop closed, recreate client for next retry
            if "Event loop is closed" in str(exc) or "closed loop" in str(exc).lower():
                global _http_client, _client_loop
                _http_client = None
                _client_loop = None
                client = _get_http_client()

        if attempt < max_retries:
            wait = min(base_wait * (2 ** (attempt - 1)), 8)  # 2s, 4s -> cap 8s
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
# Sync wrapper (for handlers.py, dictionary extraction, and other sync callers)
# ---------------------------------------------------------------------------


def call_llm_sync(
    prompt: str,
    model: str = None,
    json_format: bool = False,
    temperature: float = None,
) -> str:
    """
    Synchronous wrapper around call_llm_async.
    Safe to call from ThreadPoolExecutor workers or synchronous graph nodes.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coro = call_llm_async(prompt, model, json_format, temperature)

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=180)
    else:
        return asyncio.run(coro)


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

_vlm_semaphore: Optional[ThreadSafeSemaphore] = None
_vlm_http_client: Optional[httpx.AsyncClient] = None


def _get_vlm_semaphore() -> ThreadSafeSemaphore:
    """Return (or create) the VLM-specific concurrency semaphore."""
    global _vlm_semaphore
    if _vlm_semaphore is None:
        from config import settings

        concurrency = int(getattr(settings, "vlm_server_count", 1)) * int(
            getattr(settings, "vlm_max_concurrency_per_server", 1)
        )
        concurrency = max(1, concurrency)
        _vlm_semaphore = ThreadSafeSemaphore(concurrency)
        logger.info(f"[VLMClient] ThreadSafeSemaphore created: max_concurrent={concurrency}")
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
