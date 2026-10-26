import re
import logging

logger = logging.getLogger(__name__)


def _strip_think_tags(text: str) -> str:
    """Remove Qwen3 <think>...</think> chain-of-thought blocks from LLM output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _call_llm(prompt: str, model: str = None, json_format: bool = False) -> str:
    """
    Synchronous LLM call � delegates to llm_client.call_llm_sync().

    Provides:
      - Shared httpx keep-alive connection pool
      - asyncio.Semaphore(LLM_MAX_CONCURRENT=2)  concurrency cap
      - Token-bucket rate limiter
      - Per-attempt timeout + exponential-backoff retry
      - Structured latency observability (sem_wait_ms, llm_elapsed_ms)

    Returns a stripped response string. Never raises.
    """
    try:
        from agents.llm_client import call_llm_sync
        return call_llm_sync(prompt, model=model, json_format=json_format)
    except Exception as exc:
        logger.error(f"[LLM] call_llm_sync failed: {exc}", exc_info=True)
        return "{}" if json_format else f"[LLM Error: {exc}]"


async def call_llm_async(
    prompt: str,
    model: str = None,
    json_format: bool = False,
) -> str:
    """
    Async LLM call � delegates to llm_client.call_llm_async().
    Use this from async graph nodes (neaps_answer_node, executive_summary_node).
    """
    from agents.llm_client import call_llm_async as _async_impl
    return await _async_impl(prompt, model=model, json_format=json_format)
 