"""
Job Handlers - Stage 5.

Isolated handler functions for each job type. These are the exact logic that
previously lived inside Celery task functions. Each handler:

  - Accepts a plain dict payload
  - Logs start and end
  - Catches all exceptions and re-raises so WorkerManager can mark the job
  - Returns a status dict on success

Handlers must be importable in a ProcessPoolExecutor subprocess (i.e., no
module-level side-effects that require a running event loop or FastAPI context).
"""

import asyncio
import concurrent.futures
import inspect
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: safe async runner (works inside thread or subprocess)
# ---------------------------------------------------------------------------

def _run_async(coro) -> Any:
    """
    Run a coroutine safely from a sync context.

    Handles the case where an event loop is already running (e.g., inside a
    ThreadPoolExecutor that shares a loop with FastAPI).
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an already-running loop - use a sub-thread to avoid deadlock
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=600)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Handler 1: excel_ingestion
# ---------------------------------------------------------------------------

def handle_excel_ingestion(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse an Excel or CSV file and enqueue per-row json_fetch jobs.

    Payload keys:
        filepath (str): Absolute path to .xlsx or .csv file
    """
    filepath = payload.get("filepath", "")
    logger.info(f"[Handler:excel_ingestion] START filepath={filepath}")
    try:
        from ingestion.excel_csv_reader import ExcelCSVIngestionManager
        mgr    = ExcelCSVIngestionManager()
        result = mgr.process_file(Path(filepath))
        logger.info(
            f"[Handler:excel_ingestion] END filepath={filepath} "
            f"dispatched={result.get('dispatched', 0)} errors={len(result.get('errors', []))}"
        )
        return {"status": "SUCCESS", **result}
    except Exception as exc:
        logger.error(f"[Handler:excel_ingestion] FAILED filepath={filepath}: {exc}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Handler 2: json_fetch
# ---------------------------------------------------------------------------

def handle_json_fetch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch JSON from an announcement URL and ingest into Qdrant announcements_json.

    Payload keys: full IngestionRow.to_dict() dict (row_id, url, company_symbol, ...)
    """
    row_id = payload.get("row_id", "unknown")
    url    = payload.get("url", "")
    logger.info(f"[Handler:json_fetch] START row_id={row_id} url={url[:80]}")

    from ingestion.oracle_tracker import OracleTracker
    tracker = OracleTracker()

    try:
        tracker.update_status(row_id, "PROCESSING")

        # Guard: payload must have a 'url' key -- if missing the Oracle row
        # was stored with a corrupt/empty payload (e.g. from a failed migration).
        # Fail immediately as FAILED (non-retriable) rather than crashing.
        if not payload.get("url"):
            msg = f"[Handler:json_fetch] Invalid payload -- missing 'url' key (payload keys: {list(payload.keys())})"
            logger.error(msg)
            tracker.update_status(row_id, "FAILED", "missing_url_in_payload")
            return {"status": "FAILED", "error": "missing_url_in_payload", "row_id": row_id}

        from ingestion.json_processor import JsonIngestionProcessor
        processor = JsonIngestionProcessor()
        result    = processor.process_url(payload)


        if result.get("status") == "SUCCESS":
            tracker.update_status(row_id, "SUCCESS")
            logger.info(
                f"[Handler:json_fetch] END SUCCESS row_id={row_id} "
                f"chunks={result.get('chunk_count', 0)}"
            )
        else:
            error_msg = result.get("error", "unknown_error")
            tracker.update_status(row_id, "FAILED", error_msg)
            logger.warning(f"[Handler:json_fetch] END FAILED row_id={row_id} reason={error_msg}")

        return result

    except Exception as exc:
        tracker.update_status(row_id, "FAILED", str(exc)[:500])
        logger.error(f"[Handler:json_fetch] FAILED row_id={row_id}: {exc}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Handler 3: document_ingestion
# ---------------------------------------------------------------------------

def handle_document_ingestion(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingest a PDF/DOCX/TXT file into the announcements_docs Qdrant collection.

    Payload keys:
        filepath (str):  Absolute path to the document
        metadata (dict): Optional extra metadata
    """
    filepath = payload.get("filepath", "")
    metadata = payload.get("metadata", {}) or {}
    logger.info(f"[Handler:document_ingestion] START filepath={filepath}")
    try:
        from core.ingestion import IngestionPipeline
        pipeline = IngestionPipeline()
        coro = pipeline.process_document(file_path=Path(filepath), metadata=metadata)

        if inspect.iscoroutine(coro):
            result = _run_async(coro)
        else:
            result = coro

        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        chunk_count = len(chunks) if isinstance(chunks, list) else chunks
        logger.info(
            f"[Handler:document_ingestion] END SUCCESS filepath={filepath} chunks={chunk_count}"
        )
        return {"status": "SUCCESS", "filepath": filepath, "chunks": chunk_count}


    except Exception as exc:
        logger.error(f"[Handler:document_ingestion] FAILED filepath={filepath}: {exc}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Handler 4: vlm_image_processing
# ---------------------------------------------------------------------------

def handle_vlm_image_processing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single image through the VLM (Ollama vision model).

    Uses the dedicated VLM semaphore and HTTP client from agents.llm_client
    (NOT the text LLM semaphore  fully isolated concurrency).

    Payload keys:
        image_b64    (str)   base64-encoded PNG image
        image_hash   (str)   pre-computed MD5 hash of the normalised image
        page_number  (int)   1-indexed page number inside the PDF
        image_index  (int)   1-indexed image number on that page
        ann_seq      (int)   Oracle ANN_SEQ for cross-referencing
        doc_id       (str)   Unique document identifier
        metadata     (dict)  Pass-through Oracle metadata for logging
        prompt       (str)   Optional VLM prompt override
    """
    image_b64   = payload.get("image_b64") or ""
    image_hash  = payload.get("image_hash", "")
    page_number = int(payload.get("page_number", 0))
    image_index = int(payload.get("image_index", 0))
    ann_seq     = payload.get("ann_seq", "?")
    prompt      = payload.get("prompt") or None

    log_tag = (
        f"ann_seq={ann_seq} page={page_number} idx={image_index} "
        f"hash={image_hash[:8] if image_hash else 'n/a'}"
    )
    logger.info(f"[Handler:vlm_image_processing] START {log_tag}")

    if not image_b64:
        logger.warning(
            f"[Handler:vlm_image_processing] SKIP {log_tag}  missing image_b64"
        )
        return {"status": "SKIPPED", "reason": "missing_image_b64", **payload}

    try:
        # 1. Check global cache first  avoid redundant VLM inference
        from vlm.image_deduplicator import get_global_vlm_cache
        cache  = get_global_vlm_cache()
        cached = cache.get(image_hash) if image_hash else None

        if cached is not None:
            logger.info(
                f"[Handler:vlm_image_processing] CACHE HIT {log_tag} "
                f"({len(cached)} chars)  skipping VLM call"
            )
            return {
                "status":      "SUCCESS",
                "description": cached,
                "from_cache":  True,
                "ann_seq":     ann_seq,
                "page_number": page_number,
                "image_index": image_index,
                "image_hash":  image_hash,
            }

        # 2. Call VLM (sync wrapper, safe in thread executor)
        from agents.llm_client import call_vlm_sync

        description = call_vlm_sync(
            images   = [image_b64],
            prompt   = prompt or (
                "You are analyzing an image from a financial announcement PDF. "
                "Describe all visible text, numbers, charts, or tables relevant to "
                "the company disclosure. Be concise."
            ),
            metadata = {"ann_seq": ann_seq},
        )

        # 3. Store in global cache for future de-duplication
        if image_hash and description and not description.startswith("[VLM Error"):
            cache.put(image_hash, description)

        is_error = description.startswith("[VLM Error")
        status   = "FAILED" if is_error else "SUCCESS"
        logger.info(
            f"[Handler:vlm_image_processing] END {status} "
            f"{log_tag} chars={len(description)}"
        )
        return {
            "status":      status,
            "description": description,
            "from_cache":  False,
            "ann_seq":     ann_seq,
            "page_number": page_number,
            "image_index": image_index,
            "image_hash":  image_hash,
        }

    except Exception as exc:
        logger.error(
            f"[Handler:vlm_image_processing] FAILED {log_tag}: {exc}", exc_info=True
        )
        raise


# ---------------------------------------------------------------------------
# Handler 5: neaps_ingestion
# ---------------------------------------------------------------------------

def handle_neaps_ingestion(payload: dict) -> dict:
    """
    Ingest all NEAPS announcements for a given company symbol.
    """
    symbol = payload.get("symbol", "")
    if not symbol:
        raise ValueError("Missing 'symbol' in payload")
        
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"[Handler:neaps_ingestion] START symbol={symbol}")
    
    try:
        from ingestion.neaps_ingestion_service import run_ingestion
        stats = run_ingestion(symbol)
        
        # Generate report + send email (non-fatal)
        try:
            from ingestion.neaps_report_emailer import generate_and_send_report
            generate_and_send_report(symbol, stats)
        except Exception as report_exc:
            logger.warning(f"[Handler:neaps_ingestion] Report/email failed (non-fatal): {report_exc}")

        logger.info(f"[Handler:neaps_ingestion] END SUCCESS symbol={symbol}")
        return {"status": "SUCCESS", "stats": stats}
    except Exception as exc:
        logger.error(f"[Handler:neaps_ingestion] FAILED symbol={symbol}: {exc}", exc_info=True)
        raise


