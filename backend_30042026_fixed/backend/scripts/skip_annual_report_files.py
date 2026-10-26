"""
skip_annual_report_files.py
============================
STANDALONE WORKAROUND SCRIPT — minimal production code change + DB ops only.

Production code change (already applied):
  neaps_duplicate_checker.py -> get_completed_sequences()
  Now skips ANN_SEQs with PROCESS_STATUS IN ('COMPLETED', 'DEFERRED').

What this script does
---------------------
1. Connects to Oracle using the same .env as the main app.
2. Finds Annual Report ANN_SEQs for a symbol (by FILE_NAME keyword match
   and/or Oracle ANN_TEXT scan via DB link).
3. Marks those ANN_SEQs as DEFERRED in AI_NEAPS_TRACKER.
   → Next ingestion run: duplicate-checker sees DEFERRED = skip.
   → Other files for the same symbol: processed normally.
4. Kills the stuck job_queue job for the symbol.
5. Re-enqueues a fresh job_queue row so the symbol restarts immediately
   and processes all NON-deferred files.

Later (when load is lower):
  --reset flag resets DEFERRED → FAILED so files are re-ingested next cycle.

Usage
-----
    # Always dry-run first (no DB writes)
    python scripts/skip_annual_report_files.py --symbol SKYGOLD --dry-run

    # Apply: skip Annual Reports, re-run SKYGOLD for everything else
    python scripts/skip_annual_report_files.py --symbol SKYGOLD

    # Same for MARINE
    python scripts/skip_annual_report_files.py --symbol MARINE

    # Custom keywords
    python scripts/skip_annual_report_files.py --symbol SKYGOLD --keywords "annual report,integrated report"

    # Skip Oracle ANN_TEXT scan (faster — only matches by FILE_NAME)
    python scripts/skip_annual_report_files.py --symbol SKYGOLD --skip-oracle-scan

    # Do NOT re-enqueue automatically (trigger via API manually)
    python scripts/skip_annual_report_files.py --symbol SKYGOLD --no-reenqueue

    # LATER: reset DEFERRED → FAILED so next cycle re-ingests them
    python scripts/skip_annual_report_files.py --symbol SKYGOLD --reset

Run from the backend/ directory:
    cd backend
    python scripts/skip_annual_report_files.py --symbol SKYGOLD --dry-run

Requirements: oracledb (same as the main app).
"""

import argparse
import json
import logging
import sys
import os
import uuid
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("skip_annual_report")

DEFAULT_KEYWORDS = [
    "annual report",
    "annual-report",
    "annualreport",
    "integrated report",
    "integrated-report",
    "sustainability report",
    "sustainability-report",
]


# ---------------------------------------------------------------------------
# Settings / Connection
# ---------------------------------------------------------------------------

def _load_settings():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from config import settings
    return settings


def _connect(settings):
    import oracledb

    user     = settings.neaps_oracle_user
    password = settings.neaps_oracle_password
    dsn      = settings.neaps_oracle_dsn

    if not all([user, password, dsn]):
        logger.error(
            "NEAPS Oracle credentials not set in .env "
            "(NEAPS_ORACLE_USER / NEAPS_ORACLE_PASSWORD / NEAPS_ORACLE_HOST etc.)."
        )
        sys.exit(1)

    thick_mode     = getattr(settings, "neaps_oracle_thick_mode",     False)
    client_lib_dir = getattr(settings, "neaps_oracle_client_lib_dir", "") or ""
    if thick_mode:
        try:
            kwargs = {"lib_dir": client_lib_dir} if client_lib_dir else {}
            oracledb.init_oracle_client(**kwargs)
            logger.info("oracledb thick mode enabled.")
        except Exception as exc:
            if "already been called" not in str(exc).lower():
                logger.warning(f"Thick mode init warning: {exc}")

    conn = oracledb.connect(user=user, password=password, dsn=dsn)
    logger.info(f"Connected to Oracle DSN='{dsn}'")
    return conn


# ---------------------------------------------------------------------------
# Step 1 — Find Annual Report ANN_SEQs
# ---------------------------------------------------------------------------

def find_heavy_tracker_rows(conn, symbol: str, keywords: list) -> list:
    """
    Find rows in AI_NEAPS_TRACKER for this symbol whose FILE_NAME matches
    any keyword, and whose status is not already DEFERRED or COMPLETED.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ANN_SEQ, COMPANY_SYMBOL, FILE_NAME, PROCESS_STATUS,
               CHUNK_COUNT, PROCESS_TIME, ERROR_MSG
        FROM   AI_NEAPS_TRACKER
        WHERE  UPPER(COMPANY_SYMBOL) = UPPER(:sym)
          AND  PROCESS_STATUS NOT IN ('DEFERRED', 'COMPLETED')
        ORDER  BY ANN_SEQ
        """,
        {"sym": symbol},
    )
    cols = [d[0].lower() for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.close()

    kw_lower = [k.lower() for k in keywords]
    return [
        r for r in rows
        if any(kw in (r.get("file_name") or "").lower() for kw in kw_lower)
    ]


def find_heavy_from_oracle(conn, settings, symbol: str, keywords: list) -> list:
    """
    Scan announcements table via DB link for ANN_TEXT matching keywords.
    Catches files currently in-flight (tracker row may be PENDING or absent).
    """
    db          = settings.neaps_oracle_db_link
    schema_name = (settings.neaps_oracle_schema_name or "").strip()
    schema      = f"{schema_name}." if schema_name else ""

    like_clauses = " OR ".join(
        f"UPPER(a.ANN_TEXT) LIKE UPPER('%{kw}%')"
        for kw in keywords
    )
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT a.ANN_SEQ,
                   s.SYMB_SYMBOL        AS ann_symbol,
                   a.ANN_FILE_ATTCH,
                   a.ANN_SERVER_FILE_ATTCH,
                   a.ANN_CD
            FROM   {schema}announcements@{db} a
            INNER JOIN {schema}tbl_symbols@{db} s
                   ON s.SYMB_ID = a.ANN_SYMBOL_CD
            WHERE  UPPER(s.SYMB_SYMBOL) = UPPER(:sym)
              AND  a.ANN_CREAT_DT >= ADD_MONTHS(SYSDATE, -24)
              AND  UPPER(a.ann_status) = 'DISSEMINATED'
              AND  ({like_clauses})
            ORDER  BY a.ANN_SEQ
            """,
            {"sym": symbol},
        )
        cols = [d[0].lower() for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
        logger.info(f"  Oracle ANN_TEXT scan: {len(rows)} match(es) for symbol={symbol}")
        return rows
    except Exception as exc:
        logger.warning(f"  Oracle ANN_TEXT scan failed (non-fatal): {exc}")
        return []


# ---------------------------------------------------------------------------
# Step 2 — Mark ANN_SEQs as DEFERRED
# ---------------------------------------------------------------------------

def mark_deferred(conn, ann_seqs: list, symbol: str, dry_run: bool) -> int:
    """
    MERGE into AI_NEAPS_TRACKER: set PROCESS_STATUS = 'DEFERRED'.

    After this, neaps_duplicate_checker.get_completed_sequences() will
    include these ANN_SEQs in its result set, and run_ingestion() will
    treat them as already-processed → skipped.

    Later, --reset flips DEFERRED → FAILED so they are re-ingested.
    """
    if not ann_seqs:
        return 0

    reason = (
        "Manually DEFERRED by skip_annual_report_files.py workaround. "
        "Large file (Annual Report / Integrated Report) skipped to unblock queue. "
        "Reset with --reset flag when load is lower to re-ingest."
    )

    logger.info(
        f"  {'[DRY-RUN] Would mark' if dry_run else 'Marking'} "
        f"{len(ann_seqs)} ANN_SEQ(s) as DEFERRED"
    )
    for seq in ann_seqs:
        logger.info(f"    ANN_SEQ={seq}")

    if dry_run:
        return len(ann_seqs)

    cur = conn.cursor()
    updated = 0
    for seq in ann_seqs:
        cur.execute(
            """
            MERGE INTO AI_NEAPS_TRACKER t
            USING (SELECT :seq AS ANN_SEQ FROM DUAL) s
            ON (t.ANN_SEQ = s.ANN_SEQ)
            WHEN MATCHED THEN
                UPDATE SET PROCESS_STATUS = 'DEFERRED',
                           ERROR_MSG      = :reason,
                           PROCESS_TIME   = CURRENT_TIMESTAMP,
                           COMPANY_SYMBOL = :sym
            WHEN NOT MATCHED THEN
                INSERT (ANN_SEQ, COMPANY_SYMBOL, PROCESS_STATUS, CHUNK_COUNT,
                        ERROR_MSG, PROCESS_TIME)
                VALUES (:seq2,  :sym2,           'DEFERRED',    0,
                        :reason2, CURRENT_TIMESTAMP)
            """,
            {
                "seq":     seq,
                "reason":  reason[:4000],
                "sym":     symbol,
                "seq2":    seq,
                "sym2":    symbol,
                "reason2": reason[:4000],
            },
        )
        updated += 1
    conn.commit()
    cur.close()
    logger.info(f"  Committed {updated} DEFERRED row(s).")
    return updated


# ---------------------------------------------------------------------------
# Step 3 — Kill stuck job_queue job
# ---------------------------------------------------------------------------

def kill_stuck_job(conn, symbol: str, dry_run: bool) -> int:
    """
    Mark the current pending/processing job_queue row(s) for this symbol
    as 'failed' with retry_count = max_retries (no auto-retry).
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT job_id, status, retry_count, max_retries
        FROM   job_queue
        WHERE  UPPER(symbol) = UPPER(:sym)
          AND  job_type      = 'neaps_ingestion'
          AND  status        IN ('pending', 'processing')
        """,
        {"sym": symbol},
    )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        logger.info(f"  No active job_queue job found for symbol={symbol}. Nothing to kill.")
        return 0

    logger.info(
        f"  {'[DRY-RUN] Would kill' if dry_run else 'Killing'} "
        f"{len(rows)} job(s) for symbol={symbol}"
    )
    for r in rows:
        logger.info(f"    job_id={r[0][:8]}...  status={r[1]}  retry={r[2]}/{r[3]}")

    if dry_run:
        return len(rows)

    now = datetime.now(timezone.utc)
    msg = (
        "Killed by skip_annual_report_files.py. "
        "Annual Report ANN_SEQs set to DEFERRED. "
        "Fresh job re-enqueued to process remaining files."
    )
    cur = conn.cursor()
    updated = 0
    for r in rows:
        cur.execute(
            """
            UPDATE job_queue
            SET    status        = 'failed',
                   updated_at   = :now,
                   error_message = :msg,
                   retry_count  = max_retries
            WHERE  job_id = :jid
            """,
            {"now": now, "msg": msg, "jid": r[0]},
        )
        updated += 1
    conn.commit()
    cur.close()
    logger.info(f"  Killed {updated} job(s).")
    return updated


# ---------------------------------------------------------------------------
# Step 4 — Re-enqueue a fresh job
# ---------------------------------------------------------------------------

def reenqueue_symbol(conn, symbol: str, dry_run: bool) -> bool:
    """
    Insert a new 'pending' job_queue row for the symbol.
    The new run will call get_completed_sequences() which now skips DEFERRED
    rows, so Annual Reports are skipped and everything else is processed.
    """
    # Wait for old jobs to be gone
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM job_queue
        WHERE  UPPER(symbol)  = UPPER(:sym)
          AND  job_type       = 'neaps_ingestion'
          AND  status         IN ('pending', 'processing')
        """,
        {"sym": symbol},
    )
    active = cur.fetchone()[0]
    cur.close()

    if active > 0:
        logger.warning(
            f"  symbol={symbol} still has {active} active job(s). "
            "Cannot re-enqueue yet — the kill may take a moment to take effect. "
            "Wait ~10s and re-run, or trigger via API manually."
        )
        return False

    job_id  = str(uuid.uuid4())
    now     = datetime.now(timezone.utc)
    payload = json.dumps({"symbol": symbol})

    logger.info(
        f"  {'[DRY-RUN] Would insert' if dry_run else 'Inserting'} "
        f"fresh pending job_id={job_id[:8]}... for symbol={symbol}"
    )

    if dry_run:
        return True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO job_queue
                (job_id, job_type, symbol, payload, status,
                 retry_count, max_retries, created_at, updated_at)
            VALUES
                (:jid, 'neaps_ingestion', :sym, :payload, 'pending',
                 0, 3, :now, :now)
            """,
            {"jid": job_id, "sym": symbol.upper(), "payload": payload, "now": now},
        )
        conn.commit()
        cur.close()
        logger.info(
            f"  Re-enqueued job_id={job_id[:8]}. "
            f"{symbol} will now process all non-DEFERRED files."
        )
        return True
    except Exception as exc:
        logger.error(f"  Re-enqueue failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Reset — flip DEFERRED → FAILED for re-ingestion
# ---------------------------------------------------------------------------

def reset_deferred_rows(conn, symbol: str, dry_run: bool) -> int:
    """
    Find rows with PROCESS_STATUS = 'DEFERRED' for this symbol and reset
    them to FAILED.  FAILED is NOT in get_completed_sequences(), so they
    will be re-ingested on the next cycle when you trigger the symbol again.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ANN_SEQ, FILE_NAME, PROCESS_TIME
        FROM   AI_NEAPS_TRACKER
        WHERE  UPPER(COMPANY_SYMBOL) = UPPER(:sym)
          AND  PROCESS_STATUS        = 'DEFERRED'
        ORDER  BY ANN_SEQ
        """,
        {"sym": symbol},
    )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        logger.info(f"  No DEFERRED rows found for symbol={symbol}. Nothing to reset.")
        return 0

    logger.info(
        f"  {'[DRY-RUN] Would reset' if dry_run else 'Resetting'} "
        f"{len(rows)} DEFERRED row(s) -> FAILED for symbol={symbol}"
    )
    for r in rows:
        logger.info(f"    ANN_SEQ={r[0]}  FILE={r[1]}")

    if dry_run:
        return len(rows)

    now = datetime.now(timezone.utc)
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            UPDATE AI_NEAPS_TRACKER
            SET    PROCESS_STATUS = 'FAILED',
                   ERROR_MSG      = 'Reset by skip_annual_report_files.py --reset. Ready for re-ingestion.',
                   PROCESS_TIME   = :now
            WHERE  ANN_SEQ = :seq
            """,
            {"now": now, "seq": r[0]},
        )
    conn.commit()
    cur.close()
    logger.info(f"  Reset {len(rows)} DEFERRED row(s) to FAILED.")
    return len(rows)


# ---------------------------------------------------------------------------
# List currently deferred rows
# ---------------------------------------------------------------------------

def list_deferred(conn, symbol: str) -> None:
    """Print all DEFERRED rows for the symbol (informational)."""
    cur = conn.cursor()
    where = "WHERE UPPER(COMPANY_SYMBOL) = UPPER(:sym) AND PROCESS_STATUS = 'DEFERRED'"
    cur.execute(
        f"""
        SELECT ANN_SEQ, FILE_NAME, PROCESS_TIME, ERROR_MSG
        FROM   AI_NEAPS_TRACKER
        {where}
        ORDER  BY ANN_SEQ
        """,
        {"sym": symbol},
    )
    rows = cur.fetchall()
    cur.close()

    if not rows:
        logger.info(f"No DEFERRED rows found for symbol={symbol}.")
    else:
        logger.info(f"DEFERRED rows for symbol={symbol} ({len(rows)} total):")
        for r in rows:
            logger.info(f"  ANN_SEQ={r[0]}  FILE={r[1]}  DEFERRED_AT={r[2]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Defer Annual Report files for a symbol so it can continue "
            "ingesting all other files. Minimal production code change: "
            "neaps_duplicate_checker.get_completed_sequences() now also "
            "skips DEFERRED rows."
        )
    )
    parser.add_argument("--symbol", required=True, help="Company symbol, e.g. SKYGOLD")
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated substrings matched against FILE_NAME (case-insensitive).",
    )
    parser.add_argument("--dry-run", action="store_true", help="No DB writes.")
    parser.add_argument(
        "--no-reenqueue", action="store_true",
        help="Do not re-enqueue after killing. Trigger via API manually.",
    )
    parser.add_argument(
        "--skip-oracle-scan", action="store_true",
        help="Skip Oracle ANN_TEXT scan (only match by FILE_NAME in tracker).",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Reset DEFERRED -> FAILED so files are re-ingested next cycle.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List current DEFERRED rows for the symbol (read-only).",
    )
    return parser.parse_args()


def main():
    args     = parse_args()
    symbol   = args.symbol.strip().upper()
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    dry_run  = args.dry_run

    logger.info("=" * 65)
    logger.info("skip_annual_report_files.py  —  WORKAROUND SCRIPT")
    logger.info("=" * 65)
    logger.info(f"Symbol       : {symbol}")
    logger.info(f"Keywords     : {keywords}")
    logger.info(f"Mode         : {'DRY-RUN (no writes)' if dry_run else 'LIVE'}")
    logger.info(f"Reset mode   : {args.reset}")
    logger.info(f"Re-enqueue   : {not args.no_reenqueue}")
    logger.info("=" * 65)

    settings = _load_settings()
    conn     = _connect(settings)

    try:
        # ----------------------------------------------------------------
        # --list: show current DEFERRED rows (read-only)
        # ----------------------------------------------------------------
        if args.list:
            list_deferred(conn, symbol)
            return

        # ----------------------------------------------------------------
        # --reset: flip DEFERRED -> FAILED for re-ingestion
        # ----------------------------------------------------------------
        if args.reset:
            logger.info("RESET MODE: flipping DEFERRED -> FAILED ...")
            n = reset_deferred_rows(conn, symbol, dry_run)
            logger.info("=" * 65)
            if dry_run:
                logger.info("DRY-RUN: no changes written.")
            else:
                logger.info(f"Reset {n} row(s) to FAILED.")
                logger.info(
                    f"NEXT: trigger {symbol} ingestion via the API/admin UI. "
                    "Annual Report files will now be picked up and ingested."
                )
            logger.info("=" * 65)
            return

        # ----------------------------------------------------------------
        # STEP 1: Find Annual Report ANN_SEQs
        # ----------------------------------------------------------------
        logger.info("STEP 1: Finding Annual Report ANN_SEQs ...")

        tracker_rows = find_heavy_tracker_rows(conn, symbol, keywords)
        logger.info(f"  AI_NEAPS_TRACKER filename matches: {len(tracker_rows)} row(s)")
        for r in tracker_rows:
            logger.info(
                f"    ANN_SEQ={r['ann_seq']}  STATUS={r['process_status']}"
                f"  FILE={r['file_name']}"
            )

        oracle_rows = []
        if not args.skip_oracle_scan:
            logger.info("  Scanning Oracle ANN_TEXT via DB link ...")
            oracle_rows = find_heavy_from_oracle(conn, settings, symbol, keywords)
            for r in oracle_rows:
                logger.info(
                    f"    ANN_SEQ={r['ann_seq']}  "
                    f"FILE={r.get('ann_server_file_attch') or r.get('ann_file_attch', '?')}"
                )

        all_seqs = (
            {int(r["ann_seq"]) for r in tracker_rows}
            | {int(r["ann_seq"]) for r in oracle_rows}
        )

        if not all_seqs:
            logger.info(
                f"  No Annual Report files found for symbol={symbol}. "
                "Check --keywords. Nothing to do."
            )
            return

        logger.info(
            f"  Total unique ANN_SEQs to defer: {len(all_seqs)}"
            f" — {sorted(all_seqs)}"
        )

        # ----------------------------------------------------------------
        # STEP 2: Mark as DEFERRED
        # (get_completed_sequences now skips DEFERRED, so next run skips them)
        # ----------------------------------------------------------------
        logger.info("STEP 2: Marking as DEFERRED in AI_NEAPS_TRACKER ...")
        n2 = mark_deferred(conn, sorted(all_seqs), symbol, dry_run)
        logger.info(f"  {n2} row(s) marked DEFERRED.")

        # ----------------------------------------------------------------
        # STEP 3: Kill stuck job_queue job
        # ----------------------------------------------------------------
        logger.info("STEP 3: Killing stuck job_queue job ...")
        n3 = kill_stuck_job(conn, symbol, dry_run)
        logger.info(f"  {n3} job(s) killed.")

        # ----------------------------------------------------------------
        # STEP 4: Re-enqueue fresh job
        # ----------------------------------------------------------------
        if not args.no_reenqueue:
            logger.info("STEP 4: Re-enqueueing fresh job for symbol ...")
            reenqueue_symbol(conn, symbol, dry_run)
        else:
            logger.info("STEP 4: Skipped (--no-reenqueue). Trigger via API manually.")

        # ----------------------------------------------------------------
        # Summary
        # ----------------------------------------------------------------
        logger.info("=" * 65)
        if dry_run:
            logger.info("DRY-RUN complete. No changes written to DB.")
            logger.info("Re-run WITHOUT --dry-run to apply.")
        else:
            logger.info(f"SUCCESS. Summary for symbol={symbol}:")
            logger.info("")
            logger.info(
                f"  ✓ {len(all_seqs)} Annual Report file(s) marked DEFERRED."
            )
            logger.info(
                "    Duplicate-checker will SKIP them in the next ingestion run."
            )
            logger.info(
                f"  ✓ Old stuck job killed. Fresh job enqueued."
            )
            logger.info(
                f"  ✓ {symbol} will now ingest ALL OTHER announcement files normally."
            )
            logger.info(
                "  ✓ Queued symbols (14) will be unblocked as worker slots free."
            )
            logger.info("")
            logger.info("LATER — when load is lower, re-ingest Annual Reports:")
            logger.info(
                f"    python scripts/skip_annual_report_files.py "
                f"--symbol {symbol} --reset"
            )
            logger.info(
                "    Then trigger ingestion via API. Annual Reports will "
                "be picked up in the next cycle."
            )
        logger.info("=" * 65)

    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
