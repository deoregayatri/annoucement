"""
Oracle DB tracker for Excel/CSV ingestion pipeline -- Stage 2.

Tracks ingestion state per URL row using Oracle DB (oracledb library).

Tables managed:
  - excel_ingestion_log   : one row per URL with status / error tracking

Connection settings pulled from .env:
  ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN
  (e.g. ORACLE_DSN=localhost:1521/XEPDB1)
"""

import logging
from datetime import datetime
from typing import Optional, Any

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_DDL_EXCEL_INGESTION_LOG = """
CREATE TABLE excel_ingestion_log (
    id              VARCHAR2(36)   PRIMARY KEY,
    url_hash        VARCHAR2(64)   NOT NULL,
    url             VARCHAR2(2000) NOT NULL,
    company_name    VARCHAR2(255)  NOT NULL,
    company_symbol  VARCHAR2(50)   NOT NULL,
    title           VARCHAR2(1000),
    platform        VARCHAR2(255),
    mentioned_date  VARCHAR2(50),
    status          VARCHAR2(50)   DEFAULT 'PENDING' NOT NULL,
    error_msg       CLOB,
    created_at      TIMESTAMP      DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at      TIMESTAMP      DEFAULT CURRENT_TIMESTAMP NOT NULL
)
"""

_IDX_URL_HASH = "CREATE INDEX idx_excel_url_hash ON excel_ingestion_log(url_hash)"


class OracleTracker:
    """
    Thin Oracle DB wrapper for tracking Excel/CSV ingestion rows.

    Connection is lazy -- first call to any method triggers connect().
    Falls back to a no-op stub when Oracle is not configured, so the
    app can still boot without a DB.
    """

    def __init__(self) -> None:
        self._conn: Optional[Any] = None
        self._available: Optional[bool] = None  # None = not yet tested
        logger.info("[OracleTracker] Initialised (lazy connect).")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> Optional[Any]:
        """Return a live Oracle connection, or None if unavailable."""
        if self._available is False:
            return None
        if self._conn is not None:
            try:
                self._conn.ping()
                return self._conn
            except Exception:
                logger.warning("[OracleTracker] Connection lost, reconnecting ...")
                self._conn = None

        try:
            import oracledb
            user     = getattr(settings, "oracle_user",     None)
            password = getattr(settings, "oracle_password", None)
            dsn      = getattr(settings, "oracle_dsn",      None)

            if not all([user, password, dsn]):
                logger.warning(
                    "[OracleTracker] ORACLE_USER / ORACLE_PASSWORD / ORACLE_DSN not set in .env. "
                    "Running in no-op mode -- ingestion still works but tracking is disabled."
                )
                self._available = False
                return None

            # ------------------------------------------------------------------
            # Thick mode: required for NNE (Native Network Encryption),
            # which is the default on corporate / enterprise Oracle servers.
            # Set ORACLE_THICK_MODE=true in .env to enable.
            # ------------------------------------------------------------------
            thick_mode     = getattr(settings, "oracle_thick_mode",     False)
            client_lib_dir = getattr(settings, "oracle_client_lib_dir", "") or ""
            if thick_mode:
                try:
                    init_kwargs = {"lib_dir": client_lib_dir} if client_lib_dir else {}
                    oracledb.init_oracle_client(**init_kwargs)
                    logger.info("[OracleTracker] oracledb thick mode enabled.")
                except Exception as thick_exc:
                    # init_oracle_client raises if already called -- safe to ignore
                    if "already been called" not in str(thick_exc).lower():
                        logger.warning(f"[OracleTracker] Thick mode init warning: {thick_exc}")

            self._conn = oracledb.connect(user=user, password=password, dsn=dsn)
            self._available = True
            logger.info(f"[OracleTracker] Connected to Oracle DSN='{dsn}'")
            self._ensure_tables()
            return self._conn

        except ImportError:
            logger.warning("[OracleTracker] oracledb not installed -- running in no-op mode.")
            self._available = False
            return None
        except Exception as exc:
            logger.error(f"[OracleTracker] Oracle connection failed: {exc} -- running in no-op mode.")
            self._available = False
            return None

    def _ensure_tables(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        conn = self._conn
        if conn is None:
            return
        cur = conn.cursor()
        # Check if table exists
        cur.execute(
            "SELECT COUNT(*) FROM user_tables WHERE table_name = 'EXCEL_INGESTION_LOG'"
        )
        if cur.fetchone()[0] == 0:
            try:
                cur.execute(_DDL_EXCEL_INGESTION_LOG)
                cur.execute(_IDX_URL_HASH)
                conn.commit()
                logger.info("[OracleTracker] Created excel_ingestion_log table and index.")
            except Exception as exc:
                logger.error(f"[OracleTracker] Table creation failed: {exc}")
        cur.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def url_hash_exists(self, url_hash: str) -> bool:
        """Return True if this URL hash is already in the log."""
        conn = self._get_conn()
        if conn is None:
            return False  # No tracking -> allow through
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM excel_ingestion_log WHERE url_hash = :1",
                [url_hash]
            )
            exists = cur.fetchone()[0] > 0
            cur.close()
            return exists
        except Exception as exc:
            logger.error(f"[OracleTracker] url_hash_exists error: {exc}")
            return False

    def insert_row(self, row: Any) -> None:
        """
        Insert a new IngestionRow into the tracking table with status=PENDING.

        Args:
            row: IngestionRow instance (has .to_dict())
        """
        conn = self._get_conn()
        if conn is None:
            return
        try:
            d = row.to_dict() if hasattr(row, "to_dict") else row
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO excel_ingestion_log
                  (id, url_hash, url, company_name, company_symbol, title, platform,
                   mentioned_date, status, created_at, updated_at)
                VALUES
                  (:1, :2, :3, :4, :5, :6, :7, :8, 'PENDING', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                [
                    d["row_id"],
                    d["url_hash"],
                    d["url"],
                    d["company_name"],
                    d["company_symbol"],
                    d["title"],
                    d.get("platform", ""),
                    d.get("mentioned_date", ""),
                ]
            )
            conn.commit()
            cur.close()
            logger.debug(f"[OracleTracker] Inserted row_id={d['row_id']}")
        except Exception as exc:
            logger.error(f"[OracleTracker] insert_row failed: {exc}")

    def update_status(self, row_id: str, status: str, error_msg: str = "") -> None:
        """
        Update row status after fetch/processing attempt.

        Args:
            row_id: UUID of the row
            status: 'SUCCESS' | 'FAILED' | 'PROCESSING'
            error_msg: Error details on failure
        """
        conn = self._get_conn()
        if conn is None:
            return
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE excel_ingestion_log
                SET status = :1, error_msg = :2, updated_at = CURRENT_TIMESTAMP
                WHERE id = :3
                """,
                [status, error_msg or None, row_id]
            )
            conn.commit()
            cur.close()
        except Exception as exc:
            logger.error(f"[OracleTracker] update_status failed: {exc}")

    def get_stats(self) -> dict:
        """Return ingestion log statistics."""
        conn = self._get_conn()
        if conn is None:
            return {"oracle_available": False}
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT status, COUNT(*) FROM excel_ingestion_log
                GROUP BY status
                """
            )
            rows = cur.fetchall()
            cur.close()
            return {
                "oracle_available": True,
                "by_status": {r[0]: r[1] for r in rows},
            }
        except Exception as exc:
            logger.error(f"[OracleTracker] get_stats failed: {exc}")
            return {"oracle_available": False, "error": str(exc)}

    def close(self) -> None:
        """Close Oracle connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
