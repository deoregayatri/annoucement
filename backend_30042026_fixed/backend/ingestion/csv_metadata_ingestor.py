"""
CSVMetadataIngestor — orchestrates CSV parsing, Oracle insert, company matching,
and Qdrant ingestion via DescriptionIndexer.

Changes:
  - All print() calls replaced with news.ingestion logger calls.
  - Every step (file skip, URL dedup, Oracle insert, match, index) is logged
    to logs/news_ingestion.log via the "news.ingestion" logger.
  - No changes to business logic.
"""

import logging
import os
import csv
import shutil
from datetime import datetime

from ingestion.company_matcher import CompanyMatcher
from ingestion.description_indexer import DescriptionIndexer

# All ingestion events go to news_ingestion.log
logger = logging.getLogger("news.ingestion")


class CSVMetadataIngestor:

    def __init__(self, db_connection):
        self.conn    = db_connection
        self.matcher = None   # Lazy init
        self.indexer = None   # Lazy init

    # ------------------------------------------------------------------
    # CHECK FILE PROCESSED
    # ------------------------------------------------------------------

    def is_file_processed(self, file_name: str) -> bool:
        query = """
        SELECT 1 FROM DWM_ANN_NEWS
        WHERE FILE_NAME = :file_name
        FETCH FIRST 1 ROWS ONLY
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, {"file_name": file_name})
            return cursor.fetchone() is not None

    # ------------------------------------------------------------------
    # CHECK URL EXISTS
    # ------------------------------------------------------------------

    def record_exists(self, url: str) -> bool:
        query = """
        SELECT 1 FROM DWM_ANN_NEWS
        WHERE URL = :url
        FETCH FIRST 1 ROWS ONLY
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, {"url": url})
            return cursor.fetchone() is not None

    # ------------------------------------------------------------------
    # INSERT RECORD
    # ------------------------------------------------------------------

    def insert_record(self, file_name: str, url: str, company_name: str):
        query = """
        INSERT INTO DWM_ANN_NEWS
        (FILE_NAME, URL, COMPANY_NAME, INSERTED_AT)
        VALUES (:file_name, :url, :company_name, :inserted_at)
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, {
                "file_name":    file_name,
                "url":          url,
                "company_name": company_name,
                "inserted_at":  datetime.now(),
            })

    # ------------------------------------------------------------------
    # MOVE FILE
    # ------------------------------------------------------------------

    def move_file(self, source_path: str, destination_dir: str, skipped: bool = False):
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        original_name = os.path.basename(source_path)
        name, ext     = os.path.splitext(original_name)
        new_name      = f"{name}_skipped{ext}" if skipped else original_name
        dest          = os.path.join(destination_dir, new_name)
        shutil.move(source_path, dest)
        logger.info("File moved → %s", dest)

    # ------------------------------------------------------------------
    # PROCESS SINGLE CSV FILE
    # ------------------------------------------------------------------

    def process_csv_file(self, file_path: str) -> bool:
        file_name = os.path.basename(file_path)
        logger.info("=== Processing file: %s ===", file_name)

        rows_total    = 0
        rows_skipped  = 0
        rows_inserted = 0
        rows_matched  = 0
        rows_indexed  = 0
        rows_no_match = 0

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader)
                header = [h.strip().lower() for h in header]

                if "company name" not in header or "url" not in header:
                    logger.error(
                        "Required columns missing in %s — found: %s", file_name, header
                    )
                    raise Exception("Required columns missing")

                company_idx = header.index("company name")
                url_idx     = header.index("url")

                for row_num, row in enumerate(reader, start=2):
                    rows_total += 1

                    if len(row) < len(header):
                        logger.warning("Skipping bad row %d in %s", row_num, file_name)
                        rows_skipped += 1
                        continue

                    company_name = str(row[company_idx]).strip()
                    url          = str(row[url_idx]).strip()

                    if not url:
                        logger.debug("Row %d has empty URL — skipping", row_num)
                        rows_skipped += 1
                        continue

                    # ── URL dedup ─────────────────────────────────────────
                    if self.record_exists(url):
                        logger.info(
                            "Duplicate URL skipped (row %d): %s", row_num, url
                        )
                        rows_skipped += 1
                        continue

                    # ── Oracle insert ─────────────────────────────────────
                    self.insert_record(file_name, url, company_name)
                    rows_inserted += 1
                    logger.info(
                        "Oracle INSERT row=%d company=%r url=%s",
                        row_num, company_name, url,
                    )

                    # ── Company matcher (lazy load) ────────────────────────
                    if self.matcher is None:
                        logger.info("Initializing CompanyMatcher (lazy)")
                        from ingestion.company_matcher import CompanyMatcher
                        self.matcher = CompanyMatcher()

                    matches = self.matcher.match(company_name)

                    if matches:
                        best            = matches[0]
                        matched_company = best["matched_name"]
                        symbol          = best["symbol"]
                        rows_matched   += 1
                        logger.info(
                            "MATCH row=%d csv_name=%r → matched=%r symbol=%s "
                            "type=%s score=%.2f",
                            row_num, company_name, matched_company, symbol,
                            best.get("type"), best.get("score", 0),
                        )

                        # ── DescriptionIndexer (lazy load) ─────────────────
                        if self.indexer is None:
                            logger.info("Initializing DescriptionIndexer (lazy)")
                            from ingestion.description_indexer import DescriptionIndexer
                            self.indexer = DescriptionIndexer()

                        self.indexer.process_single(
                            url=url,
                            csv_company_name=company_name,
                            matched_company=matched_company,
                            symbol=symbol,
                        )
                        rows_indexed += 1

                    else:
                        rows_no_match += 1
                        logger.warning(
                            "NO MATCH row=%d company=%r url=%s — "
                            "skipping Qdrant indexing",
                            row_num, company_name, url,
                        )

            self.conn.commit()
            logger.info(
                "=== DONE %s — total=%d skipped=%d inserted=%d "
                "matched=%d indexed=%d no_match=%d ===",
                file_name, rows_total, rows_skipped, rows_inserted,
                rows_matched, rows_indexed, rows_no_match,
            )
            return True

        except Exception as e:
            logger.error(
                "Error processing %s: %s", file_name, str(e), exc_info=True
            )
            self.conn.rollback()
            return False

    # ------------------------------------------------------------------
    # PROCESS DIRECTORY
    # ------------------------------------------------------------------

    def process_directory(self, directory_path: str, processed_dir: str):
        logger.info(
            "process_directory START dir=%r processed_dir=%r",
            directory_path, processed_dir,
        )
        csv_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".csv")]
        logger.info("Found %d CSV file(s) in %s", len(csv_files), directory_path)

        for file in csv_files:
            full_path = os.path.join(directory_path, file)
            file_name = os.path.basename(full_path)

            # ── File-level skip ───────────────────────────────────────────
            if self.is_file_processed(file_name):
                logger.info("File already processed — skipping: %s", file_name)
                self.move_file(full_path, processed_dir, skipped=True)
                continue

            # ── Process file ──────────────────────────────────────────────
            success = self.process_csv_file(full_path)
            if success:
                self.move_file(full_path, processed_dir, skipped=False)
            else:
                logger.error(
                    "process_csv_file failed for %s — file NOT moved", file_name
                )