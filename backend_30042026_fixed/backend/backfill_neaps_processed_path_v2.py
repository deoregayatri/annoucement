"""
backfill_neaps_processed_path_v2.py
====================================

Backfills AI_NEAPS_TRACKER.PROCESSED_SFTP_PATH for existing NEAPS records
with proper path resolution using a SINGLE SFTP recursive scan.

Strategy (fast — designed for 6000+ records):
  1. Open ONE SFTP connection.
  2. Walk the entire SFTP_REMOTE_BASE once → build {filename: full_path} index.
  3. For every DB record:
       a. If ANN_SERVER_FILE_ATTCH is a full absolute path and that path is in
          the index, use it directly.
       b. Otherwise look up the bare filename in the index (O(1) dict lookup).
  4. Bulk-update AI_NEAPS_TRACKER.PROCESSED_SFTP_PATH.
  5. Commit once.

This avoids opening a new SFTP connection per record, which would make
6308 records take hours. With the single-scan approach total runtime is
typically 5–30 minutes (dominated by the one-time SFTP directory walk).

IMPORTANT:
- This script does NOT download files.
- This script does NOT run ingestion.
- This script does NOT modify PROCESS_STATUS, LOCAL_PATH, or FILE_DELETED.
- Only PROCESSED_SFTP_PATH is updated.
- Uses private key SFTP auth (same as neaps_sftp_fetcher_prod.py).
"""

import logging
import stat
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import oracledb
import paramiko

from config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("neaps_backfill_v2")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRACKER_TABLE    = "AI_NEAPS_TRACKER"
DB_LINK          = settings.neaps_oracle_db_link
SCHEMA           = (settings.neaps_oracle_schema_name or "").strip()
SCHEMA_PREFIX    = f"{SCHEMA}." if SCHEMA else ""

SFTP_HOST            = settings.neaps_sftp_host
SFTP_PORT            = settings.neaps_sftp_port
SFTP_USER            = settings.neaps_sftp_user
SFTP_PKEY_PATH       = settings.neaps_sftp_pkey_path
SFTP_PKEY_PASSPHRASE = settings.neaps_sftp_pkey_passphrase or None
SFTP_REMOTE_BASE     = settings.neaps_sftp_remote_base.rstrip("/") + "/"

# Commit to DB every N rows (reduces memory pressure, allows partial progress)
COMMIT_BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# SFTP connection  (mirrors neaps_sftp_fetcher_prod.py exactly)
# ---------------------------------------------------------------------------

def _make_sftp():
    """Open a paramiko SFTP connection using private key auth."""
    transport = paramiko.Transport((SFTP_HOST, SFTP_PORT))

    pkey_path = Path(SFTP_PKEY_PATH).expanduser()
    if not pkey_path.is_file():
        raise FileNotFoundError(f"SFTP private key not found: {pkey_path}")

    pkey = None
    last_exc = None
    for key_cls_name in ("RSAKey", "ECDSAKey", "Ed25519Key", "DSSKey"):
        key_cls = getattr(paramiko, key_cls_name, None)
        if key_cls is None:
            continue
        try:
            pkey = key_cls.from_private_key_file(
                str(pkey_path), password=SFTP_PKEY_PASSPHRASE
            )
            break
        except Exception as exc:
            last_exc = exc

    if pkey is None:
        raise RuntimeError(
            f"Could not load private key from {pkey_path}. Last error: {last_exc}"
        )

    transport.connect(username=SFTP_USER, pkey=pkey)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return transport, sftp

# ---------------------------------------------------------------------------
# SFTP single-scan index builder
# ---------------------------------------------------------------------------

def build_sftp_index(sftp, remote_dir: str, index: Dict[str, str], depth: int = 0) -> None:
    """
    Recursively walk remote_dir and populate index as {filename: full_path}.

    If the same filename appears in multiple directories, the LAST occurrence
    wins (consistent with how the old per-file recursive search worked).
    Logs progress every 100 directories scanned.
    """
    try:
        entries = sftp.listdir_attr(remote_dir)
    except UnicodeDecodeError as exc:
        logger.warning("[SFTP] Encoding issue in '%s': %s — skipping", remote_dir, exc)
        return
    except Exception as exc:
        logger.warning("[SFTP] Cannot list '%s': %s", remote_dir, exc)
        return

    for entry in entries:
        try:
            entry_path = remote_dir.rstrip("/") + "/" + entry.filename
        except UnicodeDecodeError as exc:
            logger.warning("[SFTP] Undecodable filename in '%s': %s — skipping", remote_dir, exc)
            continue

        if stat.S_ISDIR(entry.st_mode):
            build_sftp_index(sftp, entry_path, index, depth + 1)
        else:
            index[entry.filename] = entry_path


def load_sftp_index() -> Dict[str, str]:
    """
    Open ONE SFTP connection, walk SFTP_REMOTE_BASE completely, and return
    a dict mapping every filename found to its full remote path.
    """
    logger.info("[SFTP] Connecting to %s:%s as %s …", SFTP_HOST, SFTP_PORT, SFTP_USER)
    transport, sftp = _make_sftp()

    index: Dict[str, str] = {}
    t0 = time.time()

    logger.info("[SFTP] Starting full directory scan under: %s", SFTP_REMOTE_BASE)
    logger.info("[SFTP] (This is a one-time scan — may take several minutes for large trees)")

    try:
        build_sftp_index(sftp, SFTP_REMOTE_BASE, index)
    finally:
        sftp.close()
        transport.close()

    elapsed = time.time() - t0
    logger.info(
        "[SFTP] Scan complete in %.1fs — %d unique filenames indexed.",
        elapsed, len(index)
    )
    return index

# ---------------------------------------------------------------------------
# Oracle connection
# ---------------------------------------------------------------------------

def get_connection():
    """Create an Oracle connection using NEAPS settings."""
    user    = settings.neaps_oracle_user
    password = settings.neaps_oracle_password
    host    = settings.neaps_oracle_host
    port    = settings.neaps_oracle_port
    service = settings.neaps_oracle_service

    if not user:
        raise RuntimeError("NEAPS_ORACLE_USER is not configured.")
    if not password:
        raise RuntimeError("NEAPS_ORACLE_PASSWORD is not configured.")
    if not host:
        raise RuntimeError("NEAPS_ORACLE_HOST is not configured.")

    dsn = f"{host}:{port}/{service}"
    logger.info("Connecting to Oracle: %s:%s/%s", host, port, service)

    if settings.neaps_oracle_thick_mode:
        try:
            lib_dir = settings.neaps_oracle_client_lib_dir or ""
            if lib_dir:
                oracledb.init_oracle_client(lib_dir=lib_dir)
            else:
                oracledb.init_oracle_client()
            logger.info("Oracle thick mode enabled.")
        except Exception as exc:
            if "already been called" not in str(exc).lower():
                raise

    connection = oracledb.connect(user=user, password=password, dsn=dsn)
    logger.info("Oracle connection established.")
    return connection

# ---------------------------------------------------------------------------
# Fetch records
# ---------------------------------------------------------------------------

def fetch_eligible_records(connection):
    """Return list of (ann_seq, file_ref) rows that need PROCESSED_SFTP_PATH."""
    sql = f"""
        SELECT
            t.ANN_SEQ,
            a.ANN_SERVER_FILE_ATTCH
        FROM {TRACKER_TABLE} t
        INNER JOIN {SCHEMA_PREFIX}announcements@{DB_LINK} a
            ON a.ANN_SEQ = t.ANN_SEQ
        WHERE t.PROCESSED_SFTP_PATH IS NULL
          AND a.ANN_SERVER_FILE_ATTCH IS NOT NULL
    """
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        return cursor.fetchall()
    finally:
        cursor.close()

def count_eligible(connection) -> int:
    sql = f"""
        SELECT COUNT(*)
        FROM {TRACKER_TABLE} t
        INNER JOIN {SCHEMA_PREFIX}announcements@{DB_LINK} a
            ON a.ANN_SEQ = t.ANN_SEQ
        WHERE t.PROCESSED_SFTP_PATH IS NULL
          AND a.ANN_SERVER_FILE_ATTCH IS NOT NULL
    """
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        return cursor.fetchone()[0]
    finally:
        cursor.close()

# ---------------------------------------------------------------------------
# Resolve path using the pre-built index
# ---------------------------------------------------------------------------

def resolve_path(file_ref: str, sftp_index: Dict[str, str]) -> Optional[str]:
    """
    Resolve file_ref to a full SFTP path using the pre-built index.

    Priority:
      1. If file_ref is an absolute path and its basename is in the index
         at that exact path → return it.
      2. Look up the bare filename in the index (handles bare names and
         absolute paths whose exact location moved).
      3. Return None if not found.
    """
    filename = Path(file_ref).name

    # Case 1: absolute path given and it matches the indexed path exactly
    if file_ref.startswith("/") and sftp_index.get(filename) == file_ref:
        return file_ref

    # Case 2: look up by filename
    resolved = sftp_index.get(filename)
    if resolved:
        return resolved

    return None

# ---------------------------------------------------------------------------
# Backfill (uses index — no per-record SFTP calls)
# ---------------------------------------------------------------------------

def backfill_processed_path(connection, rows, sftp_index: Dict[str, str]) -> int:
    """
    Update PROCESSED_SFTP_PATH for all eligible rows using the SFTP index.
    Commits every COMMIT_BATCH_SIZE rows so partial progress is preserved.
    """
    sql_update = f"""
        UPDATE {TRACKER_TABLE}
        SET PROCESSED_SFTP_PATH = :resolved_path
        WHERE ANN_SEQ = :ann_seq
    """

    cursor = connection.cursor()
    updated_count = 0
    skipped_count = 0
    total = len(rows)

    try:
        for i, (ann_seq, file_ref) in enumerate(rows, start=1):
            resolved = resolve_path(file_ref, sftp_index)

            if resolved:
                cursor.execute(sql_update, {
                    "resolved_path": resolved,
                    "ann_seq": ann_seq,
                })
                updated_count += 1
                logger.debug("Updated ANN_SEQ %s → %s", ann_seq, resolved)
            else:
                skipped_count += 1
                logger.warning(
                    "Skipped ANN_SEQ %s — '%s' not found in SFTP index",
                    ann_seq, file_ref
                )

            # Intermediate commit + progress log every COMMIT_BATCH_SIZE rows
            if i % COMMIT_BATCH_SIZE == 0:
                connection.commit()
                logger.info(
                    "Progress: %d/%d processed (updated=%d, skipped=%d)",
                    i, total, updated_count, skipped_count
                )

        # Final commit for the remaining rows
        connection.commit()
        logger.info(
            "Backfill complete — updated: %d, skipped: %d (total: %d)",
            updated_count, skipped_count, total
        )
        return updated_count

    finally:
        cursor.close()

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_backfill(connection) -> int:
    sql = f"""
        SELECT COUNT(*)
        FROM {TRACKER_TABLE} t
        INNER JOIN {SCHEMA_PREFIX}announcements@{DB_LINK} a
            ON a.ANN_SEQ = t.ANN_SEQ
        WHERE t.PROCESSED_SFTP_PATH IS NULL
          AND a.ANN_SERVER_FILE_ATTCH IS NOT NULL
    """
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        return cursor.fetchone()[0]
    finally:
        cursor.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    connection = None
    t_start = time.time()

    try:
        logger.info("=" * 80)
        logger.info("NEAPS PROCESSED_SFTP_PATH BACKFILL V2 — single-scan mode")
        logger.info("=" * 80)
        logger.info("SFTP pkey  : %s", SFTP_PKEY_PATH or "(NOT SET — will fail)")
        logger.info("SFTP host  : %s:%s", SFTP_HOST, SFTP_PORT)
        logger.info("Remote base: %s", SFTP_REMOTE_BASE)
        logger.info("Batch size : %d rows per intermediate commit", COMMIT_BATCH_SIZE)

        # Step 1 — Oracle connect + count
        connection = get_connection()
        eligible = count_eligible(connection)
        logger.info("Records eligible for backfill: %d", eligible)

        if eligible == 0:
            logger.info("Nothing to backfill. Exiting.")
            return

        # Step 2 — Build SFTP index (one connection, one scan)
        sftp_index = load_sftp_index()
        logger.info("SFTP index ready — %d files indexed.", len(sftp_index))

        # Step 3 — Fetch eligible rows from Oracle
        rows = fetch_eligible_records(connection)
        logger.info("Fetched %d eligible rows from Oracle.", len(rows))

        # Step 4 — Resolve + update using index
        updated = backfill_processed_path(connection, rows, sftp_index)

        # Step 5 — Verify
        remaining = verify_backfill(connection)
        elapsed = time.time() - t_start

        print("\n" + "=" * 80)
        print("BACKFILL RESULT")
        print("=" * 80)
        print(f"Eligible before  : {eligible}")
        print(f"SFTP files indexed: {len(sftp_index)}")
        print(f"Rows updated     : {updated}")
        print(f"Remaining NULL   : {remaining}")
        print(f"Total time       : {elapsed:.1f}s")
        print("=" * 80)

        if remaining == 0:
            print("SUCCESS: All PROCESSED_SFTP_PATH values backfilled.")
        else:
            print(
                f"WARNING: {remaining} record(s) could not be backfilled "
                "(files not found in SFTP index)."
            )

    except Exception as exc:
        if connection:
            try:
                connection.rollback()
                logger.info("Transaction rolled back.")
            except Exception:
                pass
        logger.exception("Backfill failed: %s", exc)
        sys.exit(1)

    finally:
        if connection:
            try:
                connection.close()
                logger.info("Oracle connection closed.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
