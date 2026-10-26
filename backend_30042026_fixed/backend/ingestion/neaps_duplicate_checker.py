"""
neaps_duplicate_checker.py
===========================
Manages the AI_NEAPS_TRACKER tracking table in Oracle.

Before processing any document we check whether its ANN_SEQ already exists
in AI_NEAPS_TRACKER with COMPLETED or DEFERRED status.  If it does, we skip it.

DEFERRED is a temporary hold status used by the ops workaround script
(scripts/skip_annual_report_files.py) to defer large/slow files (e.g. Annual
Reports) without permanently excluding them.  A DEFERRED row is treated like
COMPLETED during ingestion (skipped), but can be reset to FAILED later so it
is re-ingested in a subsequent cycle.

All connection details are read from .env via settings (same NEAPS Oracle
settings used by NeapsOracleClient).

Table schema (auto-created on first connect):
  ANN_SEQ         NUMBER         PRIMARY KEY
  COMPANY_SYMBOL  VARCHAR2(100)
  FILE_NAME       VARCHAR2(500)
  LOCAL_PATH      VARCHAR2(1000)
  PROCESS_STATUS  VARCHAR2(50)   -- PENDING | COMPLETED | FAILED | SKIPPED | DEFERRED
  PROCESS_TIME    TIMESTAMP
  CHUNK_COUNT     NUMBER
  ERROR_MSG       CLOB
  FILE_DELETED    NUMBER(1)      -- 1 when local file deleted after embedding
"""

import logging
from datetime import datetime
from typing import Optional, Any, List, Dict

from config import settings

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE AI_NEAPS_TRACKER (
    ANN_SEQ         NUMBER         PRIMARY KEY,
    COMPANY_SYMBOL  VARCHAR2(100),
    FILE_NAME       VARCHAR2(500),
    LOCAL_PATH      VARCHAR2(1000),
    PROCESS_STATUS  VARCHAR2(50)   DEFAULT 'PENDING',
    PROCESS_TIME    TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    CHUNK_COUNT     NUMBER         DEFAULT 0,
    ERROR_MSG       CLOB,
    FILE_DELETED    NUMBER(1)      DEFAULT 0
)
"""

# Migration: add FILE_DELETED column if upgrading from old schema
_DDL_ADD_FILE_DELETED = """
ALTER TABLE AI_NEAPS_TRACKER ADD (FILE_DELETED NUMBER(1) DEFAULT 0)
"""


class NeapsDuplicateChecker:
    """
    Oracle-backed duplicate / status tracker for NEAPS announcements.

    Falls back to no-op mode (always reports 'not processed') when Oracle
    is unavailable, so the rest of the pipeline can still run.
    """

    def __init__(self) -> None:
        self._conn: Optional[Any] = None
        self._available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_conn(self) -> Optional[Any]:
        if self._available is False:
            return None
        if self._conn is not None:
            try:
                self._conn.ping()
                return self._conn
            except Exception:
                self._conn = None

        try:
            import oracledb

            user     = settings.neaps_oracle_user
            password = settings.neaps_oracle_password
            dsn      = settings.neaps_oracle_dsn

            if not all([user, password, dsn]):
                logger.warning("[NeapsDupChecker] Oracle creds not set - no-op mode.")
                self._available = False
                return None

            if settings.neaps_oracle_thick_mode:
                try:
                    lib_dir = settings.neaps_oracle_client_lib_dir or ""
                    kwargs = {"lib_dir": lib_dir} if lib_dir else {}
                    oracledb.init_oracle_client(**kwargs)
                except Exception as exc:
                    if "already been called" not in str(exc).lower():
                        logger.warning(f"[NeapsDupChecker] Thick mode init warning: {exc}")

            self._conn = oracledb.connect(user=user, password=password, dsn=dsn)
            self._available = True
            logger.info(f"[NeapsDupChecker] Connected to Oracle DSN='{dsn}'")
            self._ensure_table()
            return self._conn

        except ImportError:
            logger.warning("[NeapsDupChecker] oracledb not installed - no-op mode.")
            self._available = False
            return None
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] Connection failed: {exc} - no-op mode.")
            self._available = False
            return None

    def _ensure_table(self) -> None:
        """Create AI_NEAPS_TRACKER if not exists, and migrate columns if needed."""
        conn = self._conn
        if conn is None:
            return
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*) FROM user_tables WHERE table_name = 'AI_NEAPS_TRACKER'"
            )
            if cur.fetchone()[0] == 0:
                cur.execute(_DDL)
                conn.commit()
                logger.info("[NeapsDupChecker] Created AI_NEAPS_TRACKER table.")
            else:
                # Migrate: add FILE_DELETED column if missing
                cur.execute(
                    "SELECT COUNT(*) FROM user_tab_columns "
                    "WHERE table_name = 'AI_NEAPS_TRACKER' AND column_name = 'FILE_DELETED'"
                )
                if cur.fetchone()[0] == 0:
                    try:
                        cur.execute(_DDL_ADD_FILE_DELETED)
                        conn.commit()
                        logger.info("[NeapsDupChecker] Migrated AI_NEAPS_TRACKER: added FILE_DELETED column.")
                    except Exception as mig_exc:
                        logger.warning(f"[NeapsDupChecker] Migration warning (FILE_DELETED): {mig_exc}")
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] Table setup failed: {exc}")
        finally:
            cur.close()

    @staticmethod
    def _normalize_oracle_value(value: Any) -> Any:
        """
        Convert Oracle driver values (especially CLOB/BLOB wrappers) into
        JSON-serializable Python primitives.
        """
        if value is None:
            return None

        if isinstance(value, (int, float, str, bool)):
            return value

        # oracledb returns CLOB/BLOB columns as LOB objects in thick mode.
        if hasattr(value, "read") and callable(getattr(value, "read")):
            try:
                value = value.read()
            except Exception:
                pass

        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return str(value)

        if isinstance(value, (int, float, str, bool)):
            return value

        # Fallback: force any remaining weird objects to string so FastAPI JSON doesn't crash
        return str(value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_processed(self, ann_seq: int) -> bool:
        """
        Return True if ann_seq already has COMPLETED status in tracking table.
        In no-op mode always returns False (allow through).
        """
        conn = self._get_conn()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT PROCESS_STATUS FROM AI_NEAPS_TRACKER WHERE ANN_SEQ = :seq",
                {"seq": ann_seq},
            )
            row = cur.fetchone()
            cur.close()
            return row is not None and row[0] == "COMPLETED"
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] is_processed error: {exc}")
            return False

    def get_completed_sequences(self, ann_seqs: List[int]) -> set:
        """
        Bulk check: return the set of ANN_SEQs that should be SKIPPED.

        Skipped means PROCESS_STATUS is either:
          - COMPLETED : successfully ingested, never re-process.
          - DEFERRED  : temporarily held by ops workaround (e.g. large Annual
                        Report files deferred while queue is busy). Treated the
                        same as COMPLETED during ingestion. Reset to FAILED via
                        scripts/skip_annual_report_files.py --reset when ready
                        to re-ingest.

        Uses a single Oracle query with IN (...) instead of N separate queries.
        Oracle limits IN-lists to 1000 items, so we batch automatically.

        Args:
            ann_seqs: list of ANN_SEQ integers to check.

        Returns:
            set of int — ANN_SEQs that are COMPLETED or DEFERRED (skip these).
            Returns empty set in no-op mode.
        """
        if not ann_seqs:
            return set()

        conn = self._get_conn()
        if conn is None:
            return set()

        completed = set()
        try:
            cur = conn.cursor()
            # Oracle IN-list limit is 1000 items, batch accordingly
            batch_size = 999
            for i in range(0, len(ann_seqs), batch_size):
                batch = ann_seqs[i:i + batch_size]
                placeholders = ", ".join(f":s{j}" for j in range(len(batch)))
                binds = {f"s{j}": seq for j, seq in enumerate(batch)}
                cur.execute(
                    f"""
                    SELECT ANN_SEQ FROM AI_NEAPS_TRACKER
                    WHERE ANN_SEQ IN ({placeholders})
                    AND PROCESS_STATUS IN ('COMPLETED', 'DEFERRED')
                    """,
                    binds,
                )
                for row in cur.fetchall():
                    completed.add(int(row[0]))
            cur.close()
            logger.info(
                f"[NeapsDupChecker] Bulk check: {len(ann_seqs)} sequences → "
                f"{len(completed)} already COMPLETED/DEFERRED (will skip)"
            )
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] get_completed_sequences failed: {exc}")
        return completed

    def mark_pending(self, ann_seq: int, company_symbol: str, file_name: str, local_path: str) -> None:
        """Insert or update row with PENDING status."""
        conn = self._get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                """
                MERGE INTO AI_NEAPS_TRACKER t
                USING (SELECT :seq AS ANN_SEQ FROM DUAL) s
                ON (t.ANN_SEQ = s.ANN_SEQ)
                WHEN MATCHED THEN
                    UPDATE SET PROCESS_STATUS='PENDING', PROCESS_TIME=CURRENT_TIMESTAMP,
                               COMPANY_SYMBOL=:sym, FILE_NAME=:fname, LOCAL_PATH=:lpath,
                               ERROR_MSG=NULL, FILE_DELETED=0
                WHEN NOT MATCHED THEN
                    INSERT (ANN_SEQ, COMPANY_SYMBOL, FILE_NAME, LOCAL_PATH, PROCESS_STATUS,
                            PROCESS_TIME, FILE_DELETED)
                    VALUES (:seq2, :sym2, :fname2, :lpath2, 'PENDING', CURRENT_TIMESTAMP, 0)
                """,
                {
                    "seq": ann_seq, "sym": company_symbol, "fname": file_name, "lpath": local_path,
                    "seq2": ann_seq, "sym2": company_symbol, "fname2": file_name, "lpath2": local_path,
                },
            )
            conn.commit()
            cur.close()
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] mark_pending failed for ann_seq={ann_seq}: {exc}")

    def mark_completed(self, ann_seq: int, chunk_count: int = 0) -> None:
        """Update row to COMPLETED with chunk count."""
        conn = self._get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE AI_NEAPS_TRACKER
                SET PROCESS_STATUS='COMPLETED',
                    CHUNK_COUNT=:cnt,
                    PROCESS_TIME=CURRENT_TIMESTAMP
                WHERE ANN_SEQ = :seq
                """,
                {"cnt": chunk_count, "seq": ann_seq},
            )
            conn.commit()
            cur.close()
            logger.info(f"[NeapsDupChecker] Marked COMPLETED: ann_seq={ann_seq}, chunks={chunk_count}")
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] mark_completed failed for ann_seq={ann_seq}: {exc}")

    def mark_failed(self, ann_seq: int, error: str) -> None:
        """Update row to FAILED with error message."""
        conn = self._get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE AI_NEAPS_TRACKER
                SET PROCESS_STATUS='FAILED',
                    ERROR_MSG=:err,
                    PROCESS_TIME=CURRENT_TIMESTAMP
                WHERE ANN_SEQ = :seq
                """,
                {"err": str(error)[:4000], "seq": ann_seq},
            )
            conn.commit()
            cur.close()
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] mark_failed failed for ann_seq={ann_seq}: {exc}")

    def mark_skipped(self, ann_seq: int, reason: str) -> None:
        """Insert or update row with SKIPPED status (e.g. non-PDF file types)."""
        conn = self._get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                """
                MERGE INTO AI_NEAPS_TRACKER t
                USING (SELECT :seq AS ANN_SEQ FROM DUAL) s
                ON (t.ANN_SEQ = s.ANN_SEQ)
                WHEN MATCHED THEN
                    UPDATE SET PROCESS_STATUS='SKIPPED', PROCESS_TIME=CURRENT_TIMESTAMP,
                               ERROR_MSG=:reason
                WHEN NOT MATCHED THEN
                    INSERT (ANN_SEQ, PROCESS_STATUS, PROCESS_TIME, ERROR_MSG)
                    VALUES (:seq2, 'SKIPPED', CURRENT_TIMESTAMP, :reason2)
                """,
                {
                    "seq": ann_seq, "reason": str(reason)[:4000],
                    "seq2": ann_seq, "reason2": str(reason)[:4000],
                },
            )
            conn.commit()
            cur.close()
            logger.info(f"[NeapsDupChecker] Marked SKIPPED: ann_seq={ann_seq}, reason={reason}")
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] mark_skipped failed for ann_seq={ann_seq}: {exc}")

    def mark_file_deleted(self, ann_seq: int) -> None:
        """Mark the local file as deleted from disk (after successful embedding)."""
        conn = self._get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE AI_NEAPS_TRACKER SET FILE_DELETED=1 WHERE ANN_SEQ = :seq",
                {"seq": ann_seq},
            )
            conn.commit()
            cur.close()
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] mark_file_deleted failed for ann_seq={ann_seq}: {exc}")

    def get_stats(self, symbol: str = "") -> dict:
        """
        Return rich processing stats from AI_NEAPS_TRACKER.

        Returns:
            {
              'oracle_available': bool,
              'summary': {
                'total': int, 'completed': int, 'pending': int,
                'failed': int, 'skipped': int, 'total_chunks_indexed': int
              },
              'recent': [   # last 20 rows ordered by PROCESS_TIME DESC
                {
                  'ann_seq': int, 'company_symbol': str, 'file_name': str,
                  'status': str, 'chunk_count': int, 'process_time': str,
                  'error_msg': str|None, 'file_deleted': bool
                }, ...
              ]
            }
        """
        conn = self._get_conn()
        if conn is None:
            return {"oracle_available": False, "summary": {}, "recent": []}
        try:
            cur = conn.cursor()

            # Build WHERE clause
            where_clause = "WHERE UPPER(COMPANY_SYMBOL) = UPPER(:sym)" if symbol else ""
            bind = {"sym": symbol} if symbol else {}

            # Summary counts
            cur.execute(
                f"""
                SELECT
                    COUNT(*)                                       AS total,
                    SUM(CASE WHEN PROCESS_STATUS='COMPLETED' THEN 1 ELSE 0 END) AS completed,
                    SUM(CASE WHEN PROCESS_STATUS='PENDING'   THEN 1 ELSE 0 END) AS pending,
                    SUM(CASE WHEN PROCESS_STATUS='FAILED'    THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN PROCESS_STATUS='SKIPPED'   THEN 1 ELSE 0 END) AS skipped,
                    SUM(CASE WHEN PROCESS_STATUS='COMPLETED' THEN CHUNK_COUNT ELSE 0 END) AS total_chunks
                FROM AI_NEAPS_TRACKER
                {where_clause}
                """,
                bind,
            )
            row = cur.fetchone()
            summary = {
                "total":               int(row[0] or 0),
                "completed":           int(row[1] or 0),
                "pending":             int(row[2] or 0),
                "failed":              int(row[3] or 0),
                "skipped":             int(row[4] or 0),
                "total_chunks_indexed": int(row[5] or 0),
            }

            # Recent rows (last 20)
            cur.execute(
                f"""
                SELECT ANN_SEQ, COMPANY_SYMBOL, FILE_NAME, PROCESS_STATUS,
                       CHUNK_COUNT, PROCESS_TIME, ERROR_MSG, FILE_DELETED
                FROM AI_NEAPS_TRACKER
                {where_clause}
                ORDER BY PROCESS_TIME DESC
                FETCH FIRST 20 ROWS ONLY
                """,
                bind,
            )
            rows = cur.fetchall()
            cur.close()

            recent: List[Dict] = []
            for r in rows:
                process_time = r[5]
                if hasattr(process_time, "isoformat"):
                    process_time = process_time.isoformat()
                else:
                    process_time = str(process_time) if process_time else None
                recent.append({
                    "ann_seq":        int(r[0]) if r[0] is not None else None,
                    "company_symbol": r[1] or "",
                    "file_name":      r[2] or "",
                    "status":         r[3] or "",
                    "chunk_count":    int(r[4] or 0),
                    "process_time":   process_time,
                    "error_msg":      self._normalize_oracle_value(r[6]) or None,
                    "file_deleted":   bool(r[7]),
                })

            return {
                "oracle_available": True,
                "summary":          summary,
                "recent":           recent,
            }
        except Exception as exc:
            logger.error(f"[NeapsDupChecker] get_stats failed: {exc}")
            return {"oracle_available": False, "summary": {}, "recent": [], "error": str(exc)}

    def get_all_stats(self) -> dict:
        """Return global aggregate stats across all companies (no symbol filter)."""
        return self.get_stats(symbol="")

    def get_company_stats(self, symbols: Optional[List[str]] = None) -> dict:
        """
        Return per-company aggregated stats from AI_NEAPS_TRACKER.

        Args:
            symbols: Optional list of company symbols to filter by.
                     If empty/None, returns stats for all companies.

        Returns:
            {
              'oracle_available': bool,
              'rows': [
                {
                  'company_symbol': str,
                  'total':          int,   # total docs seen for this company
                  'processed':      int,   # COMPLETED
                  'skipped':        int,   # SKIPPED
                  'failed':         int,   # FAILED
                }, ...
              ]
            }
        """
        conn = self._get_conn()
        if conn is None:
            return {"oracle_available": False, "rows": []}
        try:
            cur = conn.cursor()

            if symbols:
                clean = [s.strip().upper() for s in symbols if s.strip()]
                # Oracle doesn't support bind arrays cleanly, build inline safely
                placeholders = ", ".join(f"'{s}'" for s in clean)
                where = f"WHERE UPPER(COMPANY_SYMBOL) IN ({placeholders})"
            else:
                where = ""

            cur.execute(
                f"""
                SELECT
                    COMPANY_SYMBOL,
                    COUNT(*)                                              AS total,
                    SUM(CASE WHEN PROCESS_STATUS='COMPLETED' THEN 1 ELSE 0 END) AS processed,
                    SUM(CASE WHEN PROCESS_STATUS='SKIPPED'   THEN 1 ELSE 0 END) AS skipped,
                    SUM(CASE WHEN PROCESS_STATUS='FAILED'    THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN PROCESS_STATUS IN ('PENDING', 'PROCESSING') THEN 1 ELSE 0 END) AS pending
                FROM AI_NEAPS_TRACKER
                {where}
                GROUP BY COMPANY_SYMBOL
                ORDER BY COMPANY_SYMBOL
                """
            )
            rows_raw = cur.fetchall()
            cur.close()

            rows: List[Dict] = []
            for r in rows_raw:
                raw_sym = str(r[0]).strip() if r[0] else ""
                display_sym = raw_sym if raw_sym else "Not a PDF"
                
                rows.append({
                    "company_symbol": display_sym,
                    "total":          int(r[1] or 0),
                    "processed":      int(r[2] or 0),
                    "skipped":        int(r[3] or 0),
                    "failed":         int(r[4] or 0),
                    "pending":        int(r[5] or 0),
                })

            return {"oracle_available": True, "rows": rows}

        except Exception as exc:
            logger.error(f"[NeapsDupChecker] get_company_stats failed: {exc}")
            return {"oracle_available": False, "rows": [], "error": str(exc)}

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# Singleton
_checker: Optional[NeapsDuplicateChecker] = None


def get_neaps_duplicate_checker() -> NeapsDuplicateChecker:
    global _checker
    if _checker is None:
        _checker = NeapsDuplicateChecker()
    return _checker