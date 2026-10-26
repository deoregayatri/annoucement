"""
neaps_oracle_client.py
======================
Fetches announcement metadata from the NEAPSDBA Oracle database.

All connection details are read from .env via settings:
  NEAPS_ORACLE_USER, NEAPS_ORACLE_PASSWORD,
  NEAPS_ORACLE_HOST, NEAPS_ORACLE_PORT, NEAPS_ORACLE_SERVICE
  NEAPS_ORACLE_THICK_MODE, NEAPS_ORACLE_CLIENT_LIB_DIR

Falls back to no-op mode (returns empty list) when Oracle is not reachable,
so the rest of the app can still boot without a live DB.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

from config import settings

logger = logging.getLogger(__name__)


class NeapsOracleClient:
    """
    Thin Oracle wrapper that fetches announcement records from NEAPSDBA.

    Connection is lazy -- first call to any public method triggers connect().
    """

    def __init__(self) -> None:
        self._conn: Optional[Any] = None
        self._available: Optional[bool] = None  # None = not yet tested
        logger.info("[NeapsOracleClient] Initialised (lazy connect).")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @property
    def dsn(self) -> str:
        return settings.neaps_oracle_dsn  # constructed from HOST:PORT/SERVICE

    @property
    def db_link(self) -> str:
        """Oracle DB-link name used in SQL queries (e.g. 'ANNDB')."""
        return settings.neaps_oracle_db_link

    @property
    def schema_prefix(self) -> str:
        """Oracle schema prefix (e.g. 'nsedb.'). Empty string if no schema configured."""
        s = (settings.neaps_oracle_schema_name or "").strip()
        return f"{s}." if s else ""

    def _get_conn(self) -> Optional[Any]:
        """Return a live Oracle connection, or None if unavailable."""
        if self._available is False:
            return None
        if self._conn is not None:
            try:
                self._conn.ping()
                return self._conn
            except Exception:
                logger.warning("[NeapsOracleClient] Connection lost, reconnecting...")
                self._conn = None

        try:
            import oracledb

            user = settings.neaps_oracle_user
            password = settings.neaps_oracle_password
            dsn = self.dsn

            if not all([user, password, dsn]):
                logger.warning(
                    "[NeapsOracleClient] NEAPS Oracle credentials not set in .env. "
                    "Running in no-op mode."
                )
                self._available = False
                return None

            # Thick mode for NNE-encrypted corporate Oracle
            if settings.neaps_oracle_thick_mode:
                try:
                    lib_dir = settings.neaps_oracle_client_lib_dir or ""
                    kwargs = {"lib_dir": lib_dir} if lib_dir else {}
                    oracledb.init_oracle_client(**kwargs)
                    logger.info("[NeapsOracleClient] oracledb thick mode enabled.")
                except Exception as exc:
                    if "already been called" not in str(exc).lower():
                        logger.warning(
                            f"[NeapsOracleClient] Thick mode init warning: {exc}"
                        )

            self._conn = oracledb.connect(user=user, password=password, dsn=dsn)
            self._available = True
            logger.info(f"[NeapsOracleClient] Connected to Oracle DSN='{dsn}'")
            return self._conn

        except ImportError:
            logger.warning("[NeapsOracleClient] oracledb not installed -- no-op mode.")
            self._available = False
            return None
        except Exception as exc:
            logger.error(f"[NeapsOracleClient] Connection failed: {exc} -- no-op mode.")
            self._available = False
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_announcements(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch all announcement records for a given company symbol text (e.g. 'ZOTA').

        The announcements table is accessed via DB link.
        ANN_SYMBOL_CD is a numeric FK -> tbl_symbols.symb_id;
        the human-readable symbol string is tbl_symbols.symb_symbol.
        """
        conn = self._get_conn()
        if conn is None:
            raise RuntimeError(
                f"[NeapsOracleClient] Oracle unavailable — cannot fetch announcements for '{symbol}'. "
                f"Job will be retried after back-off."
            )

        db = self.db_link
        schema = self.schema_prefix
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT a.ANN_SEQ,
                       s.SYMB_SYMBOL            AS ANN_SYMBOL_CD,
                       s.SYMB_COMPANY_NAME,
                       a.ANN_TEXT,
                       a.ANN_STATUS,
                       a.ANN_CD,
                       a.ANN_CREAT_DT,
                       a.ANN_CREAT_BY,
                       a.ANN_DT,
                       a.ANN_LST_UPD_BY,
                       a.ANN_LST_UPD_DT,
                       a.ANN_FILE_ATTCH,
                       a.ANN_SERVER_FILE_ATTCH,
                       a.ANN_RECVD_DATE,
                       a.ANN_RECVD_TIME
                FROM   {schema}announcements@{db}   a
                INNER JOIN {schema}tbl_symbols@{db} s
                       ON s.SYMB_ID = a.ANN_SYMBOL_CD
                WHERE  UPPER(s.SYMB_SYMBOL) = :symbol
                  AND  a.ANN_CREAT_DT >= ADD_MONTHS(SYSDATE, -24)
                  AND  UPPER(a.ann_status) = 'DISSEMINATED'
                  AND  UPPER(a.ANN_SERVER_FILE_ATTCH) LIKE '%PDF%'
                  AND  trunc(a.ANN_RECVD_DATE)
                       + TO_DSINTERVAL('0 ' || TRIM(a.ANN_RECVD_TIME))
                       <= systimestamp - interval '30' minute
                ORDER  BY a.ANN_DT DESC
                """,
                {"symbol": symbol.upper()},
            )
            columns = [col[0].lower() for col in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
            cur.close()
            logger.info(
                f"[NeapsOracleClient] Fetched {len(rows)} announcements for symbol='{symbol}'"
            )
            return rows
        except Exception as exc:
            logger.error(f"[NeapsOracleClient] fetch_announcements failed: {exc}")
            return []

    def fetch_announcement_by_seq(self, ann_seq: int) -> Optional[Dict[str, Any]]:
        """Fetch a single announcement record by ANN_SEQ."""
        conn = self._get_conn()
        if conn is None:
            return None
        db = self.db_link
        schema = self.schema_prefix
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT a.ANN_SEQ,
                       s.SYMB_SYMBOL       AS ANN_SYMBOL_CD,
                       s.SYMB_COMPANY_NAME,
                       a.ANN_DT,
                       a.ANN_FILE_ATTCH,
                       a.ANN_SERVER_FILE_ATTCH,
                       a.ANN_TEXT,
                       a.ANN_STATUS
                FROM   {schema}announcements@{db}   a
                INNER JOIN {schema}tbl_symbols@{db} s
                       ON s.SYMB_ID = a.ANN_SYMBOL_CD
                WHERE  a.ANN_SEQ = :seq
                """,
                {"seq": ann_seq},
            )
            columns = [col[0].lower() for col in cur.description]
            row = cur.fetchone()
            cur.close()
            return dict(zip(columns, row)) if row else None
        except Exception as exc:
            logger.error(f"[NeapsOracleClient] fetch_announcement_by_seq failed: {exc}")
            return None

    def get_unique_symbols(self) -> List[str]:
        """
        Return distinct company symbols that have DISSEMINATED announcements
        in the last 24 months.
        """
        conn = self._get_conn()
        if conn is None:
            return []
        db = self.db_link
        schema = self.schema_prefix
        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT DISTINCT symbol
                FROM (
                    SELECT s.SYMB_SYMBOL AS symbol
                    FROM   {schema}announcements@{db}   a
                    INNER JOIN {schema}tbl_symbols@{db} s
                           ON s.SYMB_ID = a.ANN_SYMBOL_CD
                    WHERE  a.ANN_CREAT_DT >= ADD_MONTHS(SYSDATE, -24)
                      AND  UPPER(a.ann_status) = 'DISSEMINATED'
                )
                ORDER BY symbol
                """
            )
            symbols = [row[0] for row in cur.fetchall()]
            cur.close()
            logger.info(
                f"[NeapsOracleClient] get_unique_symbols: {len(symbols)} DISSEMINATED symbols in last 24 months"
            )
            return symbols
        except Exception as exc:
            logger.error(f"[NeapsOracleClient] get_unique_symbols failed: {exc}")
            return []

    def close(self) -> None:
        """Close Oracle connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_client: Optional[NeapsOracleClient] = None


def get_neaps_oracle_client() -> NeapsOracleClient:
    """Get or create the NEAPS Oracle client singleton."""
    global _client
    if _client is None:
        _client = NeapsOracleClient()
    return _client
