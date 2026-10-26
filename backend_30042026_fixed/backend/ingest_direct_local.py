# -*- coding: utf-8 -*-
"""
ingest_direct_local.py
======================
Standalone ingestion script that BYPASSES the SFTP download step.

Pipeline flow:
  Oracle (Announcementss table) --> duplicate check (Qdrant ann_seq_exists)
  --> locate PDF on local mount --> text extract --> chunk --> enrich metadata
  --> hybrid embed --> Qdrant upsert

Key differences from run_neaps_ingest.py / ingest_local_neaps.py:
  * Queries the flat 'Announcementss' table directly (no @ANNDB DB-link required)
  * Uses ANN_SERVER_FILE_ATTCH (with fallback to ANN_FILE_ATTCH) as filename
  * Reads files from LOCAL_DOC_FOLDER (no SFTP)
  * No AI_NEAPS_PROCESSED tracking table - duplicate protection is done
    purely via Qdrant.ann_seq_exists() so the script is self-contained.
  * Works for ALL companies by default; use --company / --limit flags to scope.

Usage:
  # Ingest all companies found in Oracle
  python ingest_direct_local.py

  # Ingest specific companies only
  python ingest_direct_local.py --company RELIANCE TCS INFY

  # Limit rows per run (useful for smoke-tests)
  python ingest_direct_local.py --limit 50

  # Force re-ingest even if ann_seq already exists in Qdrant
  python ingest_direct_local.py --force

  # Dry-run: show what would be ingested, no writes
  python ingest_direct_local.py --dry-run

All credentials and paths are read from .env (via settings) -- nothing is
hardcoded here.  The Oracle user/password/DSN come from the NEAPS_ORACLE_*
settings block.
"""

import argparse
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_direct_local")

# ---------------------------------------------------------------------------
# Constants - override via CLI or change here
# ---------------------------------------------------------------------------
LOCAL_DOC_FOLDER = Path("/mnt/Announcement/neaps_document/Ann_docs_all_test")

# Oracle table name (flat view, no DB-link required)
ORACLE_TABLE = "Announcementss"


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------


def _init_oracle_thick(settings) -> None:
    """Initialise oracledb thick mode (required for NNE-encrypted servers)."""
    try:
        import oracledb

        lib_dir = getattr(settings, "neaps_oracle_client_lib_dir", "") or ""
        kwargs = {"lib_dir": lib_dir} if lib_dir else {}
        oracledb.init_oracle_client(**kwargs)
        logger.info("[Oracle] Thick mode initialised.")
    except Exception as exc:
        if "already been called" not in str(exc).lower():
            logger.warning(f"[Oracle] Thick mode init warning (non-fatal): {exc}")


def _connect_oracle(settings):
    """
    Return a live oracledb connection using NEAPS_ORACLE_* settings.
    Raises on failure so the caller can abort early.
    """
    import oracledb

    user = settings.neaps_oracle_user
    password = settings.neaps_oracle_password
    dsn = settings.neaps_oracle_dsn  # HOST:PORT/SERVICE constructed by settings

    if not all([user, password, dsn]):
        raise RuntimeError(
            "Oracle credentials not configured.  "
            "Set NEAPS_ORACLE_USER / NEAPS_ORACLE_PASSWORD / "
            "NEAPS_ORACLE_HOST / NEAPS_ORACLE_PORT / NEAPS_ORACLE_SERVICE in .env"
        )

    if getattr(settings, "neaps_oracle_thick_mode", False):
        _init_oracle_thick(settings)

    logger.info(f"[Oracle] Connecting to DSN='{dsn}' user='{user}' ...")
    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    logger.info("[Oracle] Connection established.")
    return conn


def fetch_all_records(
    settings,
    company_filter: Optional[List[str]] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    """
    Fetch announcement metadata from the Announcementss table.

    Args:
        settings:        App settings object (loaded from .env).
        company_filter:  Optional list of SYMB_SYMBOL values to restrict to.
                         If None/empty, ALL rows are returned.
        limit:           Maximum number of rows to return (0 = unlimited).

    Returns:
        List of row dicts with lowercase keys matching the DB column names.
    """
    conn = _connect_oracle(settings)
    try:
        cur = conn.cursor()

        # Build WHERE clause
        where_clauses = ["ANN_SERVER_FILE_ATTCH IS NOT NULL"]
        bind_vars: Dict[str, Any] = {}

        if company_filter:
            placeholders = ", ".join(f":sym{i}" for i in range(len(company_filter)))
            where_clauses.append(f"UPPER(SYMB_SYMBOL) IN ({placeholders})")
            for i, sym in enumerate(company_filter):
                bind_vars[f"sym{i}"] = sym.upper()

        where_sql = " AND ".join(where_clauses)

        # Row-limit suffix (Oracle syntax)
        limit_sql = f"AND ROWNUM <= {limit}" if limit > 0 else ""

        query = f"""
            SELECT
                ANN_SEQ,
                ANN_SYMBOL_CD,
                SYMB_COMPANY_NAME,
                ANN_DT,
                ANN_FILE_ATTCH,
                ANN_SERVER_FILE_ATTCH,
                ANN_TEXT,
                ANN_STATUS,
                ANN_CD,
                ANN_CREAT_DT,
                SYMB_SYMBOL
            FROM {ORACLE_TABLE}
            WHERE {where_sql}
            {limit_sql}
            ORDER BY ANN_SEQ DESC
        """

        logger.info(f"[Oracle] Executing query against {ORACLE_TABLE} ...")
        if bind_vars:
            cur.execute(query, bind_vars)
        else:
            cur.execute(query)

        columns = [col[0].lower() for col in cur.description]
        rows = []
        for raw_row in cur.fetchall():
            row = {}
            for col_name, val in zip(columns, raw_row):
                # Deserialise Oracle LOB / CLOBs to string
                if hasattr(val, "read"):
                    try:
                        val = val.read()
                    except Exception:
                        val = ""
                # Convert datetime to ISO string
                if hasattr(val, "isoformat"):
                    val = val.isoformat()
                row[col_name] = val
            rows.append(row)

        cur.close()
        logger.info(f"[Oracle] Fetched {len(rows)} records from {ORACLE_TABLE}.")
        return rows

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Ingestion core
# ---------------------------------------------------------------------------


def _resolve_local_path(
    row: Dict[str, Any],
    doc_folder: Path,
) -> Optional[Path]:
    """
    Determine the local file path for an announcement row.

    Priority:
      1. ANN_SERVER_FILE_ATTCH  (authoritative server filename)
      2. ANN_FILE_ATTCH         (fallback)
    """
    # Try the server filename first (as per user requirement)
    file_name = (row.get("ann_server_file_attch") or "").strip()
    if not file_name:
        file_name = (row.get("ann_file_attch") or "").strip()

    if not file_name:
        return None

    # The column may contain a full SFTP path like
    # "/home/vmuser/neapsdatafiles/file/website/ABC.pdf"
    # We only want the basename so we can look it up in the local folder.
    file_name = Path(file_name).name  # e.g. "ABC123.pdf"

    return doc_folder / file_name


def _safe_str(val: Any) -> str:
    """Convert an arbitrary Oracle value to a clean string."""
    if val is None:
        return ""
    if hasattr(val, "read"):  # LOB
        return val.read() or ""
    if hasattr(val, "isoformat"):  # datetime
        return val.isoformat()
    return str(val)


def _process_single_record_task(
    row: Dict[str, Any],
    doc_folder: Path,
    idx: int,
    total: int,
    force: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    """
    Worker function to process a single announcement record.
    Must be top-level so it can be pickled by ProcessPoolExecutor.
    """
    import logging
    import uuid
    from pathlib import Path

    # Imports must be local for cleanly spawned subprocesses
    from config import settings
    from vector_store.neaps_qdrant_store import get_neaps_qdrant_store
    from embeddings.neaps_hybrid_embedder import embed_chunks_hybrid
    from utils.document_processor import extract_pages_from_file
    from utils.text_chunker import create_chunks_from_pages_continuous
    from vlm.image_deduplicator import get_global_vlm_cache
    from vlm.vlm_pipeline import process_pdf_images, format_image_descriptions_as_text

    store = get_neaps_qdrant_store()

    ann_seq = int(row.get("ann_seq") or 0)
    company_sym = _safe_str(row.get("symb_symbol")).upper()
    company_name = _safe_str(row.get("symb_company_name"))
    prefix = f"[{idx}/{total}] ann_seq={ann_seq} company={company_sym}"

    result = {
        "status": "failed",
        "chunks": 0,
        "enriched_chunks": [],
    }

    # ------------------------------------------------------------------ #
    # 1. Duplicate check
    # ------------------------------------------------------------------ #
    if not force and store.ann_seq_exists(ann_seq):
        logger.info(f"{prefix} SKIP - already in Qdrant.")
        result["status"] = "skipped_duplicate"
        return result

    # ------------------------------------------------------------------ #
    # 2. Resolve local file path
    # ------------------------------------------------------------------ #
    local_path = _resolve_local_path(row, doc_folder)

    if local_path is None:
        logger.warning(
            f"{prefix} SKIP - no filename in ANN_SERVER_FILE_ATTCH or ANN_FILE_ATTCH."
        )
        result["status"] = "skipped_no_file"
        return result

    if not local_path.exists():
        logger.warning(f"{prefix} SKIP - file not found on disk: {local_path}")
        result["status"] = "skipped_no_file"
        return result

    # Only process PDF files
    file_ext = local_path.suffix.lower()
    if file_ext != ".pdf":
        logger.info(f"{prefix} SKIP - not a PDF (ext={file_ext}): {local_path.name}")
        result["status"] = "skipped_non_pdf"
        return result

    logger.info(f"{prefix} Processing '{local_path.name}' ...")

    # ------------------------------------------------------------------ #
    # Dry-run: stop here
    # ------------------------------------------------------------------ #
    if dry_run:
        logger.info(f"{prefix} [DRY-RUN] Would ingest '{local_path.name}'")
        result["status"] = "processed"
        return result

    # ------------------------------------------------------------------ #
    # 3. Extract & Chunk
    # ------------------------------------------------------------------ #
    try:
        file_size_mb = local_path.stat().st_size / (1024 * 1024)

        pages = extract_pages_from_file(local_path)
        if not pages:
            raise ValueError("No pages extracted from PDF.")

        raw_chunks = create_chunks_from_pages_continuous(
            pages=pages,
            filename=local_path.name,
            chunk_size=settings.neaps_chunk_size,
            chunk_overlap=settings.neaps_chunk_overlap,
            file_type="pdf",
            file_size_mb=file_size_mb,
        )

        if not raw_chunks:
            raise ValueError("Chunker produced zero chunks.")

        # ------------------------------------------------------------------ #
        # 4. Enrich chunks with Oracle metadata
        # ------------------------------------------------------------------ #
        enriched: List[Dict[str, Any]] = []
        for chunk in raw_chunks:
            meta = chunk.setdefault("metadata", {})

            # Fields required by Qdrant filter logic
            meta["company_symbol"] = company_sym
            meta["company_id"] = company_sym
            meta["active"] = True
            meta["source_type"] = "neaps_document"

            # Full Oracle metadata payload (lowercase keys for Qdrant filter compatibility)
            meta["ann_seq"] = ann_seq
            meta["ANN_SEQ"] = ann_seq
            meta["company_name"] = company_name
            meta["SYMB_SYMBOL"] = company_sym
            meta["SYMB_COMPANY_NAME"] = company_name
            meta["ANN_SYMBOL_CD"] = _safe_str(row.get("ann_symbol_cd"))
            meta["ANN_DT"] = _safe_str(row.get("ann_dt"))
            meta["ANN_CREAT_DT"] = _safe_str(row.get("ann_creat_dt"))
            meta["ANN_FILE_ATTCH"] = _safe_str(row.get("ann_file_attch"))
            meta["ANN_SERVER_FILE_ATTCH"] = _safe_str(row.get("ann_server_file_attch"))
            meta["ANN_STATUS"] = _safe_str(row.get("ann_status"))
            meta["ANN_CD"] = _safe_str(row.get("ann_cd"))

            # Unique chunk identifier
            chunk_name=f"neaps_{ann_seq}_chunk_{idx}"
            meta["chunk_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL,chunk_name))
            enriched.append(chunk)

        # ------------------------------------------------------------------ #
        # 5. Hybrid embeddings
        # ------------------------------------------------------------------ #
        texts = [c["text"] for c in enriched]
        hybrid_embs = embed_chunks_hybrid(texts)

        # ------------------------------------------------------------------ #
        # 6. Upsert to Qdrant
        # ------------------------------------------------------------------ #
        ids = store.upsert_chunks(enriched, hybrid_embs)

        logger.info(
            f"{prefix} SUCCESS - {len(ids)} chunks stored for '{local_path.name}'"
        )

        result["status"] = "processed"
        result["chunks"] = len(ids)
        result["enriched_chunks"] = enriched

        # ------------------------------------------------------------------ #
        # 7. VLM Image Processing (runs after text is already in Qdrant)
        # ------------------------------------------------------------------ #
        if (
            getattr(settings, "vlm_enabled", True)
            and local_path.suffix.lower() == ".pdf"
        ):
            try:
                ann_meta = {
                    "ann_seq": ann_seq,
                    "company_symbol": company_sym,
                    "company_name": company_name,
                }
                vlm_cache = get_global_vlm_cache()
                image_results = process_pdf_images(
                    pdf_path=local_path,
                    ann_metadata=ann_meta,
                    cache=vlm_cache,
                )

                if image_results:
                    image_text = format_image_descriptions_as_text(image_results)
                    if image_text.strip():
                        # Embed and upsert image descriptions as a separate supplemental chunk
                        supplemental_chunk = {
                            "text": image_text,
                            "metadata": {
                                "company_symbol": company_sym,
                                "company_id": company_sym,
                                "active": True,
                                "source_type": "neaps_document_images",
                                "ANN_SEQ": ann_seq,
                                "SYMB_SYMBOL": company_sym,
                                "SYMB_COMPANY_NAME": company_name,
                                "ANN_DT": _safe_str(row.get("ann_dt")),
                                "chunk_id": str(uuid.uuid4()),
                                "vlm_image_count": len(
                                    [
                                        r
                                        for r in image_results
                                        if not r.get("skipped_intra")
                                    ]
                                ),
                            },
                        }
                        supp_embs = embed_chunks_hybrid([supplemental_chunk["text"]])
                        store.upsert_chunks([supplemental_chunk], supp_embs)
                        result["vlm_images"] = len(image_results)
                        logger.info(
                            f"{prefix} VLM: {len(image_results)} images processed, "
                            f"supplemental chunk upserted ({len(image_text)} chars)"
                        )
                    else:
                        logger.info(
                            f"{prefix} VLM: {len(image_results)} images processed, "
                            f"all empty descriptions — no supplemental chunk created"
                        )

            except Exception as vlm_exc:
                # VLM failure must NEVER affect the main ingestion result
                logger.warning(
                    f"{prefix} VLM image processing failed (non-fatal): {vlm_exc}",
                    exc_info=True,
                )

    except Exception as exc:
        logger.error(f"{prefix} FAILED - '{local_path.name}': {exc}", exc_info=True)
        result["status"] = "failed"

    return result


def _subprocess_init() -> None:
    """Initializer for ProcessPoolExecutor to preload the model."""
    import logging
    import os

    pid = os.getpid()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [PID={pid}] %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Subprocess {pid}: pre-loading embedding model...")
    try:
        from core.embeddings import get_embedding_model

        # Load the SentenceTransformer model singleton into this subprocess
        get_embedding_model()
        logger.info(f"Subprocess {pid}: embedding model ready.")
    except Exception as exc:
        logger.warning(f"Subprocess {pid}: could not preload model: {exc}")


def _get_local_executor():
    """Create a standalone executor without depending on neaps_ingestion_service."""
    import os
    import concurrent.futures

    try:
        from workers.worker_manager import _compute_worker_count

        workers = _compute_worker_count()
    except Exception:
        from config import settings

        workers = getattr(settings, "max_workers_cpu", 2)

    if os.name == "nt":
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            initializer=_subprocess_init,
        )
    else:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_subprocess_init,
        )


def process_and_ingest(
    records: List[Dict[str, Any]],
    doc_folder: Path,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Process records via parallel executor to ingest directly.
    """
    import concurrent.futures

    stats: Dict[str, int] = {
        "total": len(records),
        "processed": 0,
        "skipped_duplicate": 0,
        "skipped_no_file": 0,
        "skipped_non_pdf": 0,
        "failed": 0,
        "total_chunks": 0,
    }

    all_enriched_chunks: List[Dict[str, Any]] = []

    total = len(records)
    logger.info(f"Starting parallel ingestion of {total} records from {doc_folder}")

    executor = _get_local_executor()
    futures = []

    for idx, row in enumerate(records, 1):
        future = executor.submit(
            _process_single_record_task, row, doc_folder, idx, total, force, dry_run
        )
        futures.append(future)

    for future in concurrent.futures.as_completed(futures):
        try:
            res = future.result()
            status = res.get("status", "failed")
            chunks = res.get("chunks", 0)

            if status in stats:
                stats[status] += 1
            else:
                stats[status] = stats.get(status, 0) + 1

            stats["total_chunks"] += chunks

            # Collect enriched chunks for BM25 index update
            enriched = res.get("enriched_chunks", [])
            if enriched:
                all_enriched_chunks.extend(enriched)

        except Exception as exc:
            logger.error(f"Worker exception: {exc}", exc_info=True)
            stats["failed"] += 1

    stats["_all_enriched_chunks"] = all_enriched_chunks
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    print("\n" + "=" * 60)
    print("  DIRECT LOCAL INGESTION PIPELINE (No SFTP)")
    print("  Started : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")


def _print_stats(stats: Dict[str, int], elapsed: float) -> None:
    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Total records from Oracle    : {stats['total']}")
    print(f"  Successfully ingested        : {stats['processed']}")
    print(f"  Skipped - already in Qdrant  : {stats['skipped_duplicate']}")
    print(f"  Skipped - file not found     : {stats['skipped_no_file']}")
    print(f"  Skipped - non-PDF file       : {stats['skipped_non_pdf']}")
    print(f"  Failed                       : {stats['failed']}")
    print(f"  Total new chunks stored      : {stats['total_chunks']}")
    print(f"  Elapsed time                 : {elapsed:.1f}s")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest PDFs directly from local mount into Qdrant (no SFTP)."
    )
    ap.add_argument(
        "--company",
        nargs="+",
        metavar="SYMBOL",
        help="Restrict to one or more company symbols (e.g. RELIANCE TCS).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Process at most N records from Oracle (0 = unlimited).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if ann_seq already exists in Qdrant.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without writing to Qdrant.",
    )
    ap.add_argument(
        "--doc-folder",
        default=str(LOCAL_DOC_FOLDER),
        help=(f"Local folder containing PDFs (default: {LOCAL_DOC_FOLDER})."),
    )
    args = ap.parse_args()

    _print_banner()

    doc_folder = Path(args.doc_folder)
    if not doc_folder.exists():
        logger.error(f"Document folder does not exist: {doc_folder}")
        sys.exit(1)

    from config import settings

    # -----------------------------------------------------------------------
    # 1. Fetch Oracle metadata
    # -----------------------------------------------------------------------
    company_filter = [s.upper() for s in args.company] if args.company else None
    filter_desc = ", ".join(company_filter) if company_filter else "ALL companies"
    logger.info(
        f"Fetching Oracle records for: {filter_desc} | limit={args.limit or 'none'}"
    )

    try:
        records = fetch_all_records(
            settings=settings,
            company_filter=company_filter,
            limit=args.limit,
        )
    except Exception as exc:
        logger.error(f"Failed to fetch Oracle records: {exc}", exc_info=True)
        sys.exit(1)

    if not records:
        logger.warning("No records returned from Oracle. Nothing to ingest.")
        sys.exit(0)

    if args.dry_run:
        print(f"\n[DRY-RUN MODE — no data will be written to Qdrant]\n")

    # -----------------------------------------------------------------------
    # 2. Ingest
    # -----------------------------------------------------------------------
    t0 = time.time()
    stats = process_and_ingest(
        records=records,
        doc_folder=doc_folder,
        force=args.force,
        dry_run=args.dry_run,
    )
    elapsed = time.time() - t0

    # -----------------------------------------------------------------------
    # 3. BM25 index update (main-process, after all ingestion completes)
    # -----------------------------------------------------------------------
    if not args.dry_run and stats["processed"] > 0:
        try:
            from embeddings.neaps_sparse import get_bm25_index

            bm25 = get_bm25_index()
            enriched_chunks = stats.pop("_all_enriched_chunks", [])
            if enriched_chunks:
                bm25.add_chunks(enriched_chunks, save=True)
                logger.info(f"[BM25] Index updated with {len(enriched_chunks)} chunks.")
        except Exception as bm25_exc:
            logger.warning(
                f"[BM25] Post-ingestion update failed (non-fatal): {bm25_exc}"
            )
    else:
        stats.pop("_all_enriched_chunks", None)

    _print_stats(stats, elapsed)


if __name__ == "__main__":
    main()
 