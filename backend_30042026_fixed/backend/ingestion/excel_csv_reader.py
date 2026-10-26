"""
Excel and CSV ingestion reader - Stage 2.

Handles:
  - .xlsx files via openpyxl
  - .csv files via csv stdlib

Required columns (minimum):
    company_name, title, platform, mentioned_date, url

Optional columns (passed through to metadata):
    company_symbol          - NSE/BSE ticker (derived from company_name if absent)
    description             - short description from the Excel row
    sensex_announcement     - flag column ("Yes"/"No")

Flow:
    parse file  validate rows  dedup via URL hash  dispatch job queue tasks
"""

import csv
import hashlib
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

logger = logging.getLogger(__name__)

# Only truly mandatory columns - company_symbol is optional
REQUIRED_COLUMNS = {"company_name", "title", "platform", "mentioned_date", "url"}


# ---------------------------------------------------------------------------
# Row model
# ---------------------------------------------------------------------------

class IngestionRow:
    """A single validated row ready for processing."""

    def __init__(self, raw: Dict[str, str]) -> None:
        self.company_name       = raw["company_name"].strip()
        # company_symbol is optional - fall back to empty string if absent
        self.company_symbol     = raw.get("company_symbol", "").strip().upper()
        self.title              = raw["title"].strip()
        self.platform           = raw.get("platform", "").strip()
        self.mentioned_date     = self._parse_date(raw.get("mentioned_date", ""))
        self.url                = raw["url"].strip()
        self.url_hash           = hashlib.sha256(self.url.encode()).hexdigest()
        self.row_id             = str(uuid.uuid4())
        # Additional columns captured from the Excel but not required
        self.description        = raw.get("description", "").strip()
        self.sensex_announcement = raw.get("sensex_announcement", "").strip()

    @staticmethod
    def _parse_date(raw: str) -> Optional[datetime]:
        for fmt in (
            "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%d-%b-%Y", "%d %b %Y",
            "%d %b %Y %I:%M %p",   # e.g. 24 Jun 2025 09:37 AM
            "%d %b %Y %H:%M",      # e.g. 24 Jun 2025 09:37
            "%d-%b-%Y %I:%M %p",   # e.g. 24-Jun-2025 09:37 AM
            "%d-%m-%Y %H:%M",      # e.g. 24-06-2025 09:37  (10comp.csv format)
            "%d-%m-%Y %I:%M %p",   # e.g. 24-06-2025 09:37 AM
            "%d/%m/%Y %I:%M %p",   # e.g. 24/06/2025 09:37 AM
            "%d/%m/%Y %H:%M",      # e.g. 24/06/2025 09:37
        ):
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue
        logger.warning(f"[ExcelCSVReader] Cannot parse mentioned_date='{raw}', storing as None.")
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id"              : self.row_id,
            "company_name"        : self.company_name,
            "company_symbol"      : self.company_symbol,
            "title"               : self.title,
            "platform"            : self.platform,
            "mentioned_date"      : self.mentioned_date.isoformat() if self.mentioned_date else None,
            "url"                 : self.url,
            "url_hash"            : self.url_hash,
            "description"         : self.description,
            "sensex_announcement" : self.sensex_announcement,
        }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _normalise_header(raw) -> str:
    """Lowercase + strip + underscore-normalise column names.
    Handles None keys produced by trailing commas in CSV files."""
    if raw is None:
        return ""
    return str(raw).strip().lower().replace(" ", "_").replace("-", "_")


# Aliases: map known alternate column names -> canonical required column name.
# Add more entries here as new CSV variants appear.
COLUMN_ALIASES: Dict[str, str] = {
    "mention_date"    : "mentioned_date",   # CSV typo / short form
    "date"            : "mentioned_date",
    "published_date"  : "mentioned_date",
    "pub_date"        : "mentioned_date",
    "company"         : "company_name",
    "company_nm"      : "company_name",
    "link"            : "url",
    "article_url"     : "url",
    "news_url"        : "url",
    "source_url"      : "url",
    "ticker"          : "company_symbol",
    "symbol"          : "company_symbol",
    "heading"         : "title",
    "headline"        : "title",
    "source"          : "platform",
    "media"           : "platform",
}


def _apply_aliases(key: str) -> str:
    """Resolve a normalised column name through COLUMN_ALIASES."""
    return COLUMN_ALIASES.get(key, key)


def _validate_columns(headers: List[str], filepath: str) -> None:
    norm = {_apply_aliases(_normalise_header(h)) for h in headers}
    missing = REQUIRED_COLUMNS - norm
    if missing:
        raise ValueError(
            f"[ExcelCSVReader] File '{filepath}' missing required columns: {missing}. "
            f"Found (after alias resolution): {norm}"
        )


def _normalise_row(raw_row: Dict[str, str]) -> Dict[str, str]:
    """Return a new dict with normalised + alias-resolved keys."""
    return {_apply_aliases(_normalise_header(k)): v for k, v in raw_row.items()}


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _iter_xlsx(filepath: Path) -> Iterator[Dict[str, str]]:
    """Yield row dicts from an .xlsx file."""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for .xlsx files. Install with: pip install openpyxl")

    wb = openpyxl.load_workbook(str(filepath), read_only=True, data_only=True)
    ws = wb.active

    rows = ws.iter_rows(values_only=True)
    raw_headers = [str(h) if h is not None else "" for h in next(rows)]
    _validate_columns(raw_headers, str(filepath))
    norm_headers = [_normalise_header(h) for h in raw_headers]

    for row in rows:
        values = [str(v).strip() if v is not None else "" for v in row]
        yield dict(zip(norm_headers, values))

    wb.close()


def _iter_csv(filepath: Path) -> Iterator[Dict[str, str]]:
    """Yield row dicts from a .csv file."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(filepath, "r", encoding=encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                raw_headers = reader.fieldnames or []
                _validate_columns(list(raw_headers), str(filepath))
                for row in reader:
                    yield _normalise_row(dict(row))
            return
        except UnicodeDecodeError:
            continue
    raise ValueError(f"[ExcelCSVReader] Cannot decode '{filepath}' with any supported encoding.")


# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------

class ExcelCSVIngestionManager:
    """
    Parse and validate Excel (.xlsx) or CSV (.csv) files.
    Dispatches job-queue tasks for each valid, deduplicated row.

    Usage:
        manager = ExcelCSVIngestionManager()
        summary = manager.process_file(Path("data.xlsx"))
    """

    def __init__(self) -> None:
        # Import here to avoid circular at module load
        from ingestion.oracle_tracker import OracleTracker
        self.tracker = OracleTracker()
        logger.info("[ExcelCSVIngestionManager] Initialised.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Parse file, validate rows, dedup, and dispatch job-queue tasks.

        Args:
            filepath: Path to .xlsx or .csv file

        Returns:
            Summary dict with counts
        """
        suffix = filepath.suffix.lower()
        if suffix == ".xlsx":
            row_iter = _iter_xlsx(filepath)
        elif suffix == ".csv":
            row_iter = _iter_csv(filepath)
        else:
            raise ValueError(f"[ExcelCSVIngestionManager] Unsupported file type: '{suffix}'. Use .xlsx or .csv")

        stats = {"total": 0, "skipped_invalid": 0, "skipped_duplicate": 0, "dispatched": 0, "errors": []}

        for raw in row_iter:
            stats["total"] += 1
            try:
                row = self._validate_row(raw, stats)
                if row is None:
                    continue

                # Deduplication check
                if self.tracker.url_hash_exists(row.url_hash):
                    logger.debug(f"[ExcelCSVIngestionManager] Duplicate URL skipped: {row.url[:80]}")
                    stats["skipped_duplicate"] += 1
                    continue

                # Persist to Oracle tracking table
                self.tracker.insert_row(row)

                # Dispatch job
                self._dispatch(row)
                stats["dispatched"] += 1

            except Exception as exc:
                logger.error(f"[ExcelCSVIngestionManager] Row processing error: {exc}", exc_info=True)
                stats["errors"].append(str(exc))

        logger.info(
            f"[ExcelCSVIngestionManager] File '{filepath.name}' processed - "
            f"total={stats['total']}, dispatched={stats['dispatched']}, "
            f"dup={stats['skipped_duplicate']}, invalid={stats['skipped_invalid']}"
        )
        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_row(self, raw: Dict[str, str], stats: Dict) -> Optional[IngestionRow]:
        """Validate and return an IngestionRow, or None if invalid."""
        # Check only truly required fields are non-empty
        for col in REQUIRED_COLUMNS:
            if not raw.get(col, "").strip():
                logger.warning(f"[ExcelCSVIngestionManager] Row skipped - empty required column '{col}': {raw}")
                stats["skipped_invalid"] += 1
                return None

        # URL basic sanity
        url = raw.get("url", "").strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            logger.warning(f"[ExcelCSVIngestionManager] Row skipped - invalid URL: '{url}'")
            stats["skipped_invalid"] += 1
            return None

        return IngestionRow(raw)

    @staticmethod
    def _dispatch(row: "IngestionRow") -> None:
        """Enqueue a json_fetch job into the Oracle DB job queue."""
        from ingestion.job_dispatcher import enqueue_job
        job_id = enqueue_job("json_fetch", row.to_dict(), max_retries=3)
        if job_id:
            logger.info(
                f"[ExcelCSVIngestionManager] Enqueued json_fetch job_id={job_id[:8]} "
                f"row_id={row.row_id} url={row.url[:60]}"
            )
        else:
            logger.info(
                f"[ExcelCSVIngestionManager] Ran json_fetch synchronously for row_id={row.row_id}"
            )
