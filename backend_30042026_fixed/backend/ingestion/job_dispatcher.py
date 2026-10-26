"""
Job Dispatcher - Stage 3.

Single entry point for all callers that previously used Celery task.delay() or
task.apply_async().  Replaces all of:

    from ingestion.celery_app import fetch_json_from_url
    fetch_json_from_url.apply_async(args=[row], queue="json_fetch_queue")

With:

    from ingestion.job_dispatcher import enqueue_job
    enqueue_job("json_fetch", row_dict)

If Oracle is unavailable, jobs are executed synchronously inside the calling
thread (same fallback behaviour as the old Celery stub).
"""

import logging
import threading
from typing import Any, Dict, Optional

from ingestion.job_store import get_job_store, VALID_JOB_TYPES

logger = logging.getLogger(__name__)

# Mapping job_type  sync fallback handler (used when Oracle is unavailable)
# Populated lazily behind a lock to prevent race on concurrent first calls.
_SYNC_HANDLERS: Dict[str, Any] = {}
_handlers_lock = threading.Lock()


def _get_sync_handler(job_type: str):
    """Return sync fallback callable for a given job_type (lazy import, thread-safe)."""
    if job_type in _SYNC_HANDLERS:
        return _SYNC_HANDLERS.get(job_type)
    with _handlers_lock:
        if job_type in _SYNC_HANDLERS:  # double-check inside lock
            return _SYNC_HANDLERS.get(job_type)
        try:
            from workers.handlers import (
                handle_excel_ingestion,
                handle_json_fetch,
                handle_document_ingestion,
            )

            _SYNC_HANDLERS.update(
                {
                    "excel_ingestion": handle_excel_ingestion,
                    "json_fetch": handle_json_fetch,
                    "document_ingestion": handle_document_ingestion,
                }
            )
        except Exception as exc:
            logger.warning(f"[JobDispatcher] Could not load sync handlers: {exc}")
            return None
    return _SYNC_HANDLERS.get(job_type)


def enqueue_job(
    job_type: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
) -> Optional[str]:
    """
    Enqueue a background job into the Oracle job_queue.

    Args:
        job_type:    One of 'json_fetch', 'excel_ingestion',
                     'document_ingestion', 'dictionary_extraction'
        payload:     Serialisable dict passed to the handler
        max_retries: Max retry attempts on failure (default 3)

    Raises:
        ValueError: if job_type is not in VALID_JOB_TYPES.

    Returns:
        job_id (UUID str) if Oracle is available, None in sync-fallback mode.
    """
    # Validate early - before touching DB or fallback
    if job_type not in VALID_JOB_TYPES:
        raise ValueError(
            f"[JobDispatcher] Unknown job_type={job_type!r}. "
            f"Valid: {sorted(VALID_JOB_TYPES)}"
        )

    store = get_job_store()
    job_id = store.insert_job(
        job_type=job_type, payload=payload, max_retries=max_retries
    )

    if job_id:
        logger.info(f"[JobDispatcher] Enqueued job_id={job_id[:8]} type={job_type}")
        return job_id

    # Oracle unavailable - execute synchronously as fallback
    logger.warning(
        f"[JobDispatcher] Oracle unavailable - running '{job_type}' synchronously."
    )
    handler = _get_sync_handler(job_type)
    if handler:
        try:
            handler(payload)
        except Exception as exc:
            logger.error(
                f"[JobDispatcher] Sync execution of '{job_type}' failed: {exc}",
                exc_info=True,
            )
    else:
        logger.error(
            f"[JobDispatcher] No handler found for job_type='{job_type}' - job dropped."
        )
    return None
