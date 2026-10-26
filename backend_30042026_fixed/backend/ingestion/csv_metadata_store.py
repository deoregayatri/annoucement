import logging
from typing import Optional, Any

from config import settings
import oracledb

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONNECTION (same pattern as other modules)
# ─────────────────────────────────────────────

_conn: Optional[Any] = None


def _get_conn():
    global _conn

    try:
        if _conn:
            try:
                _conn.ping()
                return _conn
            except:
                _conn = None

        # Thick mode (if required)
        if settings.neaps_oracle_thick_mode:
            try:
                oracledb.init_oracle_client(
                    lib_dir=settings.neaps_oracle_client_lib_dir
                )
            except Exception as exc:
                if "already been called" not in str(exc).lower():
                    logger.warning(f"Thick mode warning: {exc}")

        _conn = oracledb.connect(
            user=settings.neaps_oracle_user,
            password=settings.neaps_oracle_password,
            dsn=settings.neaps_oracle_dsn
        )

        return _conn

    except Exception as e:
        logger.error(f"[CSVMetadataStore] DB connection error: {e}")
        return None


# ─────────────────────────────────────────────
# STORE CLASS
# ─────────────────────────────────────────────

class CSVMetadataStore:
    """
    Stores minimal metadata for dedup:
      - company_name
      - url
    """

    TABLE_NAME = "AI_CSV_METADATA"

    # ─────────────────────────────────────────
    # CHECKS
    # ─────────────────────────────────────────

    def company_name_exists(self, company_name: str) -> bool:
        conn = _get_conn()
        if not conn:
            return False

        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT 1 FROM {self.TABLE_NAME}
            WHERE UPPER(company_name) = :name
            FETCH FIRST 1 ROWS ONLY
            """,
            {"name": company_name.upper()},
        )

        exists = cur.fetchone() is not None
        cur.close()
        return exists

    def url_exists(self, url: str) -> bool:
        conn = _get_conn()
        if not conn:
            return False

        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT 1 FROM {self.TABLE_NAME}
            WHERE url = :url
            FETCH FIRST 1 ROWS ONLY
            """,
            {"url": url},
        )

        exists = cur.fetchone() is not None
        cur.close()
        return exists

    # ─────────────────────────────────────────
    # INSERT
    # ─────────────────────────────────────────

    def insert_row(self, company_name: str, url: str) -> None:
        conn = _get_conn()
        if not conn:
            return

        try:
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT INTO {self.TABLE_NAME} (company_name, url)
                VALUES (:name, :url)
                """,
                {
                    "name": company_name,
                    "url": url,
                },
            )
            conn.commit()
            cur.close()

        except Exception as e:
            logger.error(f"[CSVMetadataStore] Insert failed: {e}")


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_store: Optional[CSVMetadataStore] = None


def get_csv_metadata_store() -> CSVMetadataStore:
    global _store
    if _store is None:
        _store = CSVMetadataStore()
    return _store