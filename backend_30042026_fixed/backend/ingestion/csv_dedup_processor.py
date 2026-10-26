"""
csv_dedup_processor.py
======================
Standalone processor that sits on top of ExcelCSVIngestionManager.

Adds cross-file deduplication using AI_CSV_METADATA before handing
valid rows to the existing ingestion pipeline.

Dedup rules (in order):
  1. company_name already in AI_CSV_METADATA → skip
  2. url already in AI_CSV_METADATA          → skip
  3. url_hash already in oracle_tracker      → skip (existing behaviour)
  4. All checks pass → insert into AI_CSV_METADATA + dispatch job

Usage:
    from ingestion.csv_dedup_processor import CsvDedupProcessor
    processor = CsvDedupProcessor()
    stats = processor.process_file(Path("comp.csv"))
"""

import csv
import hashlib
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional
from ingestion.company_matcher import CompanyMatcher


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"company_name", "title", "platform", "mentioned_date", "url"}


# ---------------------------------------------------------------------------
# Row model (same as IngestionRow in excel_csv_reader.py)
# ---------------------------------------------------------------------------

class IngestionRow:
    def __init__(self, raw: Dict[str, str]) -> None:
        self.company_name        = raw["company_name"].strip()
        self.company_symbol      = raw.get("company_symbol", "").strip().upper()
        self.title               = raw["title"].strip()
        self.platform            = raw.get("platform", "").strip()
        self.mentioned_date      = self._parse_date(raw.get("mentioned_date", ""))
        self.url                 = raw["url"].strip()
        self.url_hash            = hashlib.sha256(self.url.encode()).hexdigest()
        self.row_id              = str(uuid.uuid4())
        self.description         = raw.get("description", "").strip()
        self.sensex_announcement = raw.get("sensex_announcement", "").strip()

    @staticmethod
    def _parse_date(raw: str) -> Optional[datetime]:
        for fmt in (
            "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%d-%b-%Y", "%d %b %Y",
            "%d %b %Y %I:%M %p", "%d %b %Y %H:%M",
            "%d-%b-%Y %I:%M %p", "%d-%m-%Y %H:%M",
            "%d-%m-%Y %I:%M %p", "%d/%m/%Y %I:%M %p",
            "%d/%m/%Y %H:%M",
        ):
            try:
                return datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue
        logger.warning(f"[CsvDedupProcessor] Cannot parse date='{raw}', storing None.")
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
# Header helpers
# ---------------------------------------------------------------------------

def _norm_header(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip().lower().replace(" ", "_").replace("-", "_")


COLUMN_ALIASES: Dict[str, str] = {
    "mention_date": "mentioned_date", "date": "mentioned_date",
    "published_date": "mentioned_date", "pub_date": "mentioned_date",
    "company": "company_name", "company_nm": "company_name",
    "link": "url", "article_url": "url", "news_url": "url", "source_url": "url",
    "ticker": "company_symbol", "symbol": "company_symbol",
    "heading": "title", "headline": "title",
    "source": "platform", "media": "platform",
}


def _alias(key: str) -> str:
    return COLUMN_ALIASES.get(key, key)


def _validate_columns(headers: List[str], filepath: str) -> None:
    norm = {_alias(_norm_header(h)) for h in headers}
    missing = REQUIRED_COLUMNS - norm
    if missing:
        raise ValueError(
            f"[CsvDedupProcessor] '{filepath}' missing columns: {missing}. Found: {norm}"
        )


def _norm_row(raw_row: Dict[str, str]) -> Dict[str, str]:
    return {_alias(_norm_header(k)): v for k, v in raw_row.items()}


# ---------------------------------------------------------------------------
# Generic CSV reader (handles column shifting, BOM, bad encodings)
# ---------------------------------------------------------------------------

def _detect_encoding(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            with open(path, encoding=enc) as f:
                f.read(4096)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _is_company_header(text: str) -> bool:
    cleaned = re.sub(r"[^a-z0-9 _]", "", str(text).strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip() in ("company name", "company_name")


def _find_header_row_and_col(path: Path, encoding: str) -> Optional[tuple]:
    """Scan first 50 rows x all cols to find 'Company Name' — handles column shifting."""
    with open(path, encoding=encoding, newline="", errors="replace") as fh:
        reader = csv.reader(fh)
        for row_i, row in enumerate(reader):
            if row_i >= 50:
                break
            for col_i, cell in enumerate(row):
                if _is_company_header(cell):
                    return row_i, col_i
    return None


def _iter_csv(filepath: Path) -> Iterator[Dict[str, str]]:
    """Yield normalised row dicts; handles shifting columns, BOM, encodings."""
    encoding = _detect_encoding(filepath)
    result   = _find_header_row_and_col(filepath, encoding)

    if result is None:
        # Fallback: standard DictReader from row 0
        logger.warning(
            f"[CsvDedupProcessor] Cannot locate 'Company Name' column by scan; "
            f"falling back to DictReader for '{filepath.name}'."
        )
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                with open(filepath, "r", encoding=enc, newline="") as fh:
                    reader = csv.DictReader(fh)
                    _validate_columns(list(reader.fieldnames or []), str(filepath))
                    for row in reader:
                        yield _norm_row(dict(row))
                return
            except UnicodeDecodeError:
                continue
        raise ValueError(f"[CsvDedupProcessor] Cannot decode '{filepath}'.")

    header_row_idx, _ = result
    logger.info(
        f"[CsvDedupProcessor] '{filepath.name}': header at row {header_row_idx}."
    )

    with open(filepath, encoding=encoding, newline="", errors="replace") as fh:
        all_rows = list(csv.reader(fh))

    raw_headers   = all_rows[header_row_idx]
    headers       = [_alias(_norm_header(h.lstrip("\ufeff"))) for h in raw_headers]
    _validate_columns(headers, str(filepath))
    expected_cols = len(headers)

    for row in all_rows[header_row_idx + 1:]:
        if not any(cell.strip() for cell in row):
            continue
        if len(row) > expected_cols:
            row = row[:expected_cols - 1] + [",".join(row[expected_cols - 1:])]
        if len(row) < expected_cols:
            logger.debug(f"[CsvDedupProcessor] Short row skipped: {row}")
            continue
        yield dict(zip(headers, [v.strip() for v in row]))


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class CsvDedupProcessor:
    """
    Wraps the existing ExcelCSVIngestionManager with cross-file dedup.

    Steps per row:
      1. Validate required fields.
      2. Skip if company_name already in AI_CSV_METADATA.
      3. Skip if url already in AI_CSV_METADATA.
      4. Skip if url_hash already in oracle_tracker (existing behaviour).
      5. Insert into AI_CSV_METADATA + oracle_tracker, then dispatch job.

    Drop-in replacement for ExcelCSVIngestionManager.process_file().
    """

    def __init__(self) -> None:
        from ingestion.oracle_tracker     import OracleTracker
        from ingestion.csv_metadata_store import get_csv_metadata_store

        self.tracker    = OracleTracker()
        self.meta_store = get_csv_metadata_store()
        self.matcher = CompanyMatcher()
        logger.info("[CsvDedupProcessor] Initialised.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Process a single CSV file with full dedup logic.

        Returns:
            stats dict:
              total, skipped_invalid, skipped_dup_name, skipped_dup_url,
              skipped_dup_tracker, dispatched, errors
        """
        if filepath.suffix.lower() != ".csv":
            raise ValueError(
                f"[CsvDedupProcessor] Only .csv files supported. Got: '{filepath.suffix}'"
            )

        stats: Dict[str, Any] = {
            "total"              : 0,
            "skipped_invalid"    : 0,
            "skipped_dup_name"   : 0,   # company_name seen in AI_CSV_METADATA
            "skipped_dup_url"    : 0,   # url seen in AI_CSV_METADATA
            "skipped_dup_tracker": 0,   # url_hash seen in oracle_tracker
            "dispatched"         : 0,
            "errors"             : [],
        }

        for raw in _iter_csv(filepath):
            stats["total"] += 1
            try:
                row = self._validate_row(raw, stats)
                if row is None:
                    continue

                # ── 1. Dedup: company_name (cross-file) ───────────────────
                if self.meta_store.company_name_exists(row.company_name):
                    logger.info(
                        f"[CsvDedupProcessor] Dup company_name skipped: '{row.company_name}'"
                    )
                    stats["skipped_dup_name"] += 1
                    continue

                # ── 2. Dedup: url (cross-file) ────────────────────────────
                if self.meta_store.url_exists(row.url):
                    logger.info(
                        f"[CsvDedupProcessor] Dup URL skipped: {row.url[:80]}"
                    )
                    stats["skipped_dup_url"] += 1
                    continue

                # ── 3. Dedup: url_hash in oracle_tracker (existing) ────────
                if self.tracker.url_hash_exists(row.url_hash):
                    logger.debug(
                        f"[CsvDedupProcessor] Tracker dup skipped: {row.url[:80]}"
                    )
                    stats["skipped_dup_tracker"] += 1
                    continue

                # ── 4. All checks passed — persist + dispatch ──────────────
                self.meta_store.insert_row(row.company_name, row.url)
                self.tracker.insert_row(row)
                self._dispatch(row)
                stats["dispatched"] += 1

            except Exception as exc:
                logger.error(f"[CsvDedupProcessor] Row error: {exc}", exc_info=True)
                stats["errors"].append(str(exc))

        logger.info(
            f"[CsvDedupProcessor] '{filepath.name}' done — "
            f"total={stats['total']} dispatched={stats['dispatched']} "
            f"dup_name={stats['skipped_dup_name']} dup_url={stats['skipped_dup_url']} "
            f"dup_tracker={stats['skipped_dup_tracker']} invalid={stats['skipped_invalid']}"
        )
        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_row(self, raw: Dict[str, str], stats: Dict) -> Optional[IngestionRow]:
        for col in REQUIRED_COLUMNS:
            if not raw.get(col, "").strip():
                logger.warning(
                    f"[CsvDedupProcessor] Row skipped — empty '{col}': {raw}"
                )
                stats["skipped_invalid"] += 1
                return None

        url = raw.get("url", "").strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            logger.warning(f"[CsvDedupProcessor] Row skipped — invalid URL: '{url}'")
            stats["skipped_invalid"] += 1
            return None

        return IngestionRow(raw)

    @staticmethod
    def _dispatch(self, row):

        matches = self.matcher.match(row.company_name)

        if matches:
            best = matches[0]
            logger.info(
                f"[MATCH] {row.company_name} → {best['matched_name']} "
                f"(score={best['score']})"
            )
        else:
            logger.info(f"[MATCH] No match for {row.company_name}")


        from ingestion.job_dispatcher import enqueue_job
        job_id = enqueue_job("json_fetch", row.to_dict(), max_retries=3)
        if job_id:
            logger.info(
                f"[CsvDedupProcessor] Enqueued job_id={job_id[:8]} "
                f"row_id={row.row_id} url={row.url[:60]}"
            )
        else:
            logger.info(
                f"[CsvDedupProcessor] Ran synchronously for row_id={row.row_id}"
            )
