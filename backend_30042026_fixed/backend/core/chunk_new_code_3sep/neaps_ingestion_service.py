"""
neaps_ingestion_service.py
==========================
Orchestrates the full NEAPS ingestion pipeline for one company symbol.

Flow per company:
  1. Fetch announcement metadata from NEAPSDBA (Oracle)
  2. Filter out already-processed ANN_SEQs
  3. Parallel SFTP download of PDF files
  4. For each downloaded file (in ProcessPoolExecutor):
       a. Extract text
       b. Chunk text (800 chars / 120 overlap)
       c. Enrich chunks with Oracle metadata
       d. Generate dense embeddings (pre-loaded model in subprocess)
       e. Upsert to Qdrant in batches of 100 chunks
       f. Mark ANN_SEQ as COMPLETED in AI_NEAPS_PROCESSED
  5. BM25 update in main process (single-writer thread, non-blocking)
  6. Return stats dict with timing metrics

Process model:
  - ProcessPoolExecutor with 'spawn' context and initializer=_subprocess_init
  - _subprocess_init() loads bge-large-en-v1.5 ONCE per subprocess at startup
  - All jobs in that subprocess reuse the pre-loaded model (_subprocess_model)
  - This eliminates the 3-5s model-load overhead per document job

Observability (logged per PDF):
  - parse_ms      : time to extract pages from PDF
  - embed_ms      : time to encode all chunks
  - upsert_ms     : time to push to Qdrant
  - total_ms      : full per-PDF wall time
"""

import logging
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ============================================================================
# Subprocess model state — one per process, loaded at pool creation via init
# ============================================================================

_subprocess_model = None   # holds the pre-loaded SentenceTransformer model


def _subprocess_init() -> None:
    """
    ProcessPoolExecutor initializer — called ONCE per subprocess at pool startup.

    Loads bge-large-en-v1.5 into the subprocess's private memory space.
    All document_ingestion jobs dispatched to this process will reuse
    _subprocess_model; zero model-load cost per job after this point.

    Uses 'spawn' context (set in worker_manager.py) which guarantees:
      - Clean process state (no inherited file descriptors or locks)
      - No fork-copy quirks with CUDA/PyTorch (future-proof)
      - Consistent behaviour on Linux and Windows
    """
    global _subprocess_model
    pid = os.getpid()

    # Configure logging inside subprocess (parent handlers don't transfer with spawn)
    import logging as _logging
    _logging.basicConfig(
        level  = _logging.INFO,
        format = f"%(asctime)s [PID={pid}] %(levelname)s %(name)s: %(message)s",
    )
    _log = _logging.getLogger(__name__)

    _log.info(f"Subprocess {pid}: loading embedding model (one-time init)...")
    t0 = time.perf_counter()

    from embeddings.neaps_dense import get_dense_embedder
    _subprocess_model = get_dense_embedder()

    elapsed = (time.perf_counter() - t0) * 1000
    _log.info(f"Subprocess {pid}: embedding model ready in {elapsed:.0f}ms")


def _get_subprocess_embedder():
    """Return the pre-loaded model. Fallback-loads if somehow not initialized."""
    global _subprocess_model
    if _subprocess_model is None:
        from embeddings.neaps_dense import get_dense_embedder
        _subprocess_model = get_dense_embedder()
    return _subprocess_model


def _adaptive_embed_batch_size() -> int:
    """
    Compute safe SentenceTransformer batch size from available RAM.

    Basis: bge-large at batch=32 → ~480 MB activation peak (CPU, float32).
    Formula: batch = clamp(free_GB × 8, 8, 64)
    Cap at 64 because subprocess already holds 1.4 GB model weights.

    free_GB  → batch_size
    < 1.25   → 8   (near-OOM)
    2        → 16
    4        → 32  (normal)
    8+       → 64  (maximum safe)
    """
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1024 ** 3
        return int(max(8, min(64, free_gb * 8)))
    except Exception:
        return 32   # safe default


# ============================================================================
# Module-level reusable executor
# ============================================================================

_neaps_file_executor: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None


def _get_file_executor():
    """
    Return (or lazily create) the module-level NEAPS file-processing executor.

    On Linux: ProcessPoolExecutor with spawn + initializer (true CPU parallelism).
    On Windows: ThreadPoolExecutor with initializer (PyTorch is GIL-bound on CPU).

    Worker count is computed dynamically via _compute_worker_count() (RAM + CPU formula)
    — NOT read from MAX_WORKERS_CPU in .env, which would hardcode the static value.

    The executor is long-lived — created once, reused across all run_ingestion()
    calls. Do not create a new pool per ingestion run.
    """
    global _neaps_file_executor
    if _neaps_file_executor is not None:
        return _neaps_file_executor

    # ── Attempt to reuse WorkerManager pool (prevents duplication in FastAPI) ──
    try:
        from workers.worker_manager import get_worker_manager
        wm = get_worker_manager()
        cpu_exec = getattr(wm, "_cpu_executor", None)
        if cpu_exec is not None:
            logger.info("[NeapsIngestion] Reusing WorkerManager's CPU ProcessPoolExecutor")
            return cpu_exec
    except Exception:
        pass

    # ── Fallback: dynamic pool for standalone script runs ───────────────
    try:
        from workers.worker_manager import _compute_worker_count
        workers = _compute_worker_count()
    except Exception:
        # Fallback to settings value if worker_manager not yet importable
        from config import settings
        workers = max(1, getattr(settings, "max_workers_cpu", 2))
        logger.warning(
            f"[NeapsIngestion] Could not import _compute_worker_count; "
            f"falling back to settings.max_workers_cpu={workers}"
        )

    if os.name == "nt":   # Windows
        _neaps_file_executor = ThreadPoolExecutor(
            max_workers       = workers,
            thread_name_prefix= "neaps-file",
            initializer       = _subprocess_init,
        )
        logger.info(
            f"[NeapsIngestion] Executor: ThreadPoolExecutor "
            f"workers={workers} with model-init (Windows)"
        )
    else:   # Linux / Unix
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        _neaps_file_executor = ProcessPoolExecutor(
            max_workers = workers,
            mp_context  = ctx,
            initializer = _subprocess_init,
        )
        logger.info(
            f"[NeapsIngestion] Executor: ProcessPoolExecutor "
            f"workers={workers} spawn+initializer (Linux)"
        )
    return _neaps_file_executor



# ============================================================================
# Main ingestion orchestrator
# ============================================================================

def run_ingestion(symbol: str) -> Dict[str, Any]:
    """
    Ingest all NEAPS announcements for the given company symbol.

    Returns stats dict:
      symbol, total, skipped, downloaded, failed_dl,
      ingested, failed_proc, skipped_type, total_chunks
    """
    from ingestion.neaps_oracle_client      import get_neaps_oracle_client
    from ingestion.neaps_duplicate_checker  import get_neaps_duplicate_checker
    from ingestion.neaps_sftp_fetcher       import get_neaps_sftp_fetcher
    from vector_store.neaps_qdrant_store    import get_neaps_qdrant_store
    from config import settings
    from workers.system_state import get_system_state
    import concurrent.futures

    stats: Dict[str, Any] = {
        "symbol"      : symbol,
        "total"       : 0,
        "skipped"     : 0,
        "downloaded"  : 0,
        "failed_dl"   : 0,
        "ingested"    : 0,
        "failed_proc" : 0,
        "skipped_type": 0,
        "total_chunks": 0,
    }

    state = get_system_state()
    state.set_busy(f"ingestion:{symbol.upper()}")
    start_ts = time.time()
    logger.info(f">> START Ingestion | company={symbol.upper()}")

    try:
        # 1. Fetch metadata from NEAPSDBA
        oracle_client = get_neaps_oracle_client()
        announcements = oracle_client.fetch_announcements(symbol)
        stats["total"] = len(announcements)
        logger.info(
            f"[NeapsIngestion] company={symbol.upper()} "
            f"total={len(announcements)} announcements found"
        )

        if not announcements:
            return stats

        # 2. Filter already-processed (single bulk query instead of N separate ones)
        dup_checker = get_neaps_duplicate_checker()
        all_seqs = [int(ann.get("ann_seq", 0)) for ann in announcements]
        completed_seqs = dup_checker.get_completed_sequences(all_seqs)
        pending = [
            ann for ann in announcements
            if int(ann.get("ann_seq", 0)) not in completed_seqs
        ]
        stats["skipped"] = stats["total"] - len(pending)
        logger.info(
            f"[NeapsIngestion] company={symbol.upper()} "
            f"pending={len(pending)} skipped={stats['skipped']}"
        )
        if not pending:
            return stats

        # 3. Parallel SFTP download
        fetcher  = get_neaps_sftp_fetcher()
        store    = get_neaps_qdrant_store()
        file_map: Dict[str, Any] = {
            ann.get("ann_server_file_attch") or ann.get("ann_file_attch", ""): ann
            for ann in pending
            if ann.get("ann_server_file_attch") or ann.get("ann_file_attch")
        }
        download_results = fetcher.download_batch(list(file_map.keys()))
        stats["downloaded"] = sum(1 for v in download_results.values() if v is not None)
        stats["failed_dl"]  = sum(1 for v in download_results.values() if v is None)

        # Collect temp file paths from workers for BM25 batch update
        bm25_temp_files: List[str] = []
        bm25_batch_buffer: List[Dict[str, Any]] = []
        BM25_FLUSH_EVERY = 20  # flush BM25 buffer every N successful PDFs
        pdfs_since_flush = 0

        # 4. Process each file concurrently
        executor = _get_file_executor()
        futures  = {
            executor.submit(_process_file_task_top_level, fn, path, file_map[fn]): fn
            for fn, path in download_results.items()
        }
        for future in concurrent.futures.as_completed(futures):
            fn = futures[future]
            try:
                res = future.result()
            except Exception as worker_exc:
                logger.error(
                    f"[NeapsIngestion] company={symbol.upper()} file='{fn}' "
                    f"worker exception: {worker_exc}\n{traceback.format_exc()}"
                )
                stats["failed_proc"] += 1
                continue

            status     = res.get("status")
            dur_ms     = res.get("duration_ms", 0)
            parse_ms   = res.get("parse_ms",    0)
            embed_ms   = res.get("embed_ms",    0)
            upsert_ms  = res.get("upsert_ms",   0)

            if status == "success":
                stats["ingested"]     += 1
                stats["total_chunks"] += res["chunks"]
                # Read chunks from temp file (avoids IPC memory bloat)
                chunks_file = res.get("chunks_file")
                if chunks_file:
                    try:
                        import json as _json
                        with open(chunks_file, "r", encoding="utf-8") as cf:
                            chunk_data = _json.load(cf)
                        bm25_batch_buffer.extend(chunk_data)
                        Path(chunks_file).unlink(missing_ok=True)
                    except Exception as cf_exc:
                        logger.warning(f"[NeapsIngestion] Could not read chunks file {chunks_file}: {cf_exc}")
                pdfs_since_flush += 1
                logger.info(
                    f"[NeapsIngestion] OK | company={symbol.upper()} "
                    f"ann_seq={res['ann_seq']} file='{res['filename']}' "
                    f"chunks={res['chunks']} "
                    f"parse={parse_ms:.0f}ms embed={embed_ms:.0f}ms "
                    f"upsert={upsert_ms:.0f}ms total={dur_ms:.0f}ms"
                )
                # Incremental BM25 flush every N PDFs to prevent RAM bloat
                if pdfs_since_flush >= BM25_FLUSH_EVERY and bm25_batch_buffer:
                    try:
                        from embeddings.neaps_sparse import get_bm25_index
                        get_bm25_index().add_chunks(bm25_batch_buffer, save=False)
                        logger.info(
                            f"[NeapsIngestion] BM25 incremental flush: "
                            f"{len(bm25_batch_buffer)} chunks"
                        )
                    except Exception as bm25_exc:
                        logger.warning(f"[NeapsIngestion] BM25 flush failed (non-fatal): {bm25_exc}")
                    bm25_batch_buffer.clear()
                    pdfs_since_flush = 0
            elif status == "skipped_type":
                stats["skipped_type"] += 1
                logger.info(
                    f"[NeapsIngestion] SKIP | company={symbol.upper()} "
                    f"ann_seq={res.get('ann_seq','?')} file='{res.get('filename','?')}' "
                    f"skipped (non-PDF) total={dur_ms:.0f}ms"
                )
            elif status == "failed_proc":
                stats["failed_proc"] += 1
                logger.error(
                    f"[NeapsIngestion] FAIL | company={symbol.upper()} "
                    f"ann_seq={res.get('ann_seq','?')} file='{res.get('filename','?')}' "
                    f"failed_proc total={dur_ms:.0f}ms"
                )
            elif status == "failed_dl":
                logger.warning(
                    f"[NeapsIngestion] DL_FAIL | company={symbol.upper()} "
                    f"file='{res.get('filename','?')}' download failed"
                )

        # 5. Final BM25 flush for remaining chunks + save to disk once
        if bm25_batch_buffer:
            try:
                from embeddings.neaps_sparse import get_bm25_index
                get_bm25_index().add_chunks(bm25_batch_buffer, save=False)
            except Exception as bm25_exc:
                logger.warning(f"[NeapsIngestion] BM25 final flush failed (non-fatal): {bm25_exc}")
            bm25_batch_buffer.clear()
        # Save BM25 index to disk once at the end (not after every company)
        try:
            from embeddings.neaps_sparse import get_bm25_index
            get_bm25_index().save()
            logger.info(
                f"[NeapsIngestion] BM25 index saved to disk for {symbol.upper()}"
            )
        except Exception as bm25_save_exc:
            logger.warning(f"[NeapsIngestion] BM25 save failed (non-fatal): {bm25_save_exc}")

        elapsed_ms = (time.time() - start_ts) * 1000
        logger.info(
            f">> SUCCESS Ingestion | company={symbol.upper()} "
            f"ingested={stats['ingested']} chunks={stats['total_chunks']} "
            f"failed_proc={stats['failed_proc']} total={elapsed_ms:.0f}ms"
        )
        return stats

    except Exception as exc:
        logger.error(f">> FAILED Ingestion | company={symbol.upper()} error={exc}", exc_info=True)
        raise
    finally:
        state.clear_busy()


# ============================================================================
# Per-file worker task (runs inside subprocess)
# ============================================================================

def _process_file_task_top_level(
    filename:   str,
    local_path: Optional[Path],
    ann:        Dict[str, Any],
) -> Dict[str, Any]:
    """
    Top-level function executed by ProcessPoolExecutor workers.
    All imports are local (subprocess has a clean namespace with spawn).

    Observability:
      Returns parse_ms, embed_ms, upsert_ms, duration_ms in result dict.
    """
    pid_tag  = f"[PID={os.getpid()}]"
    t_start  = time.perf_counter()
    parse_ms = embed_ms = upsert_ms = 0.0

    from ingestion.neaps_duplicate_checker  import get_neaps_duplicate_checker
    from processing.neaps_metadata_enrichment import enrich_chunks
    from vector_store.neaps_qdrant_store    import get_neaps_qdrant_store
    from config import settings

    dup_checker = get_neaps_duplicate_checker()
    store       = get_neaps_qdrant_store()
    ann_seq     = int(ann.get("ann_seq", 0))
    ann_symbol  = (ann.get("ann_symbol_cd") or ann.get("ann_symbol") or "").upper()

    # Download failure
    if local_path is None:
        logger.warning(f"[NeapsIngestion]{pid_tag} Skipping '{filename}' — download failed.")
        dup_checker.mark_failed(ann_seq, "SFTP download failed")
        return {
            "status": "failed_dl", "ann_seq": ann_seq, "filename": filename,
            "duration_ms": (time.perf_counter() - t_start) * 1000,
        }

    # Non-PDF guard
    file_ext = Path(filename).suffix.lower()
    if file_ext != ".pdf":
        logger.info(f"{pid_tag} [NeapsIngestion] Skipping non-PDF '{filename}' (ext={file_ext})")
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass
        dup_checker.mark_skipped(ann_seq, f"Non-PDF: {file_ext}")
        return {
            "status": "skipped_type", "ann_seq": ann_seq, "filename": filename,
            "duration_ms": (time.perf_counter() - t_start) * 1000,
        }

    dup_checker.mark_pending(ann_seq, ann.get("ann_symbol_cd", ""), filename, str(local_path))

    try:
        file_size_mb = local_path.stat().st_size / (1024 * 1024)

        # ── Parse ────────────────────────────────────────────────────────────
        t_parse = time.perf_counter()
        from utils.document_processor import extract_pages_from_file
        from utils.text_chunker        import create_token_chunks_from_pages

        pages = extract_pages_from_file(local_path)
        if not pages:
            raise ValueError("No pages extracted from PDF")

        # --- Use the advanced token chunker ---
        # preserve_sections=True respects markdown and FAQ headers
        raw_chunks = create_token_chunks_from_pages(
            pages=pages,
            filename=filename,
            chunk_size=settings.neaps_chunk_size,
            overlap=settings.neaps_chunk_overlap,
            preserve_sections=True,
            file_type="pdf",
            file_size_mb=file_size_mb,
        )
        if not raw_chunks:
            raise ValueError("Chunker produced no chunks")
        parse_ms = (time.perf_counter() - t_parse) * 1000

        # ── Enrich metadata ──────────────────────────────────────────────────
        enriched = enrich_chunks(raw_chunks, ann)
        import uuid
        for c in enriched:
            c.setdefault("metadata", {})["chunk_id"] = str(uuid.uuid4())

        # ── Embed (pre-loaded model) ─────────────────────────────────────────
        t_embed     = time.perf_counter()
        model       = _get_subprocess_embedder()
        texts       = [c["text"] for c in enriched]
        batch_size  = _adaptive_embed_batch_size()

        logger.info(
            f"{pid_tag} [NeapsIngestion] Embedding {len(texts)} chunks "
            f"batch_size={batch_size} for '{filename}'"
        )
        dense_vecs = model.encode(
            texts,
            batch_size           = batch_size,
            normalize_embeddings = True,
            show_progress_bar    = False,
            convert_to_numpy     = True,
        )
        hybrid_embs = [
            {"dense": dense_vecs[i], "sparse": {"indices": [], "values": []}}
            for i in range(len(dense_vecs))
        ]
        embed_ms = (time.perf_counter() - t_embed) * 1000

        # ── Upsert to Qdrant in batches of 500 ─────────────────────────────
        t_upsert   = time.perf_counter()
        BATCH_SIZE = 500
        all_ids: List[str] = []
        for i in range(0, len(enriched), BATCH_SIZE):
            b_chunks = enriched[i:i + BATCH_SIZE]
            b_embs   = hybrid_embs[i:i + BATCH_SIZE]
            batch_ids = store.upsert_chunks(b_chunks, b_embs)
            all_ids.extend(batch_ids)
            logger.debug(
                f"{pid_tag} [NeapsIngestion] Upserted batch "
                f"{i//BATCH_SIZE + 1} ({len(b_chunks)} chunks)"
            )
        upsert_ms = (time.perf_counter() - t_upsert) * 1000

        # ── Mark completed ───────────────────────────────────────────────────
        dup_checker.mark_completed(ann_seq, chunk_count=len(all_ids))

        # ── Delete local PDF ─────────────────────────────────────────────────
        try:
            local_path.unlink(missing_ok=True)
            dup_checker.mark_file_deleted(ann_seq)
            logger.info(f"{pid_tag} [NeapsIngestion] Deleted local PDF: {local_path}")
        except Exception as del_exc:
            logger.warning(f"{pid_tag} [NeapsIngestion] Could not delete {local_path}: {del_exc}")

        # ── Save chunks to temp file (avoids IPC memory bloat) ───────────────
        import json as _json
        import tempfile as _tempfile
        chunks_file = None
        try:
            # Only save text + metadata (not embeddings) for BM25
            bm25_data = [{"text": c.get("text", ""), "metadata": c.get("metadata", {})} for c in enriched]
            fd, chunks_file = _tempfile.mkstemp(suffix=".json", prefix=f"neaps_chunks_{ann_seq}_")
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                _json.dump(bm25_data, tf)
        except Exception as tf_exc:
            logger.warning(f"{pid_tag} [NeapsIngestion] Could not write chunks temp file: {tf_exc}")
            chunks_file = None

        duration_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"{pid_tag} [NeapsIngestion] DONE | '{filename}' "
            f"chunks={len(all_ids)} "
            f"parse={parse_ms:.0f}ms embed={embed_ms:.0f}ms "
            f"upsert={upsert_ms:.0f}ms total={duration_ms:.0f}ms"
        )
        return {
            "status"     : "success",
            "ann_seq"    : ann_seq,
            "filename"   : filename,
            "chunks"     : len(all_ids),
            "chunks_file": chunks_file,   # temp file path (not the data itself)
            "parse_ms"   : parse_ms,
            "embed_ms"   : embed_ms,
            "upsert_ms"  : upsert_ms,
            "duration_ms": duration_ms,
        }

    except Exception as exc:
        duration_ms = (time.perf_counter() - t_start) * 1000
        logger.error(
            f"{pid_tag} [NeapsIngestion] FAIL | '{filename}' "
            f"ann_seq={ann_seq}: {exc} (total={duration_ms:.0f}ms)",
            exc_info=True,
        )
        dup_checker.mark_failed(ann_seq, str(exc)[:500])
        return {
            "status"     : "failed_proc",
            "ann_seq"    : ann_seq,
            "filename"   : filename,
            "duration_ms": duration_ms,
        }
