"""
file_watcher.py — watches src/ for new CSVs and triggers ingestion pipeline.
All events logged to logs/news_ingestion.log via "news.ingestion" logger.
"""

import logging
import os
import shutil
import time

from ingestion.csv_metadata_ingestor import CSVMetadataIngestor

logger = logging.getLogger("news.ingestion")

SRC_DIR       = "src"
PROCESSED_DIR = "processed"


def start_watcher():
    logger.info("File watcher started — listening for CSVs in '%s/'", SRC_DIR)

    ingestor = CSVMetadataIngestor()

    os.makedirs(SRC_DIR,       exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    while True:
        csv_files = [f for f in os.listdir(SRC_DIR) if f.endswith(".csv")]

        if csv_files:
            logger.info("Detected %d CSV file(s) in %s", len(csv_files), SRC_DIR)

        for filename in csv_files:
            file_path = os.path.join(SRC_DIR, filename)
            logger.info("Processing file: %s", filename)

            success = ingestor.process_file(file_path)

            if success:
                dest_path = os.path.join(PROCESSED_DIR, filename)
                shutil.move(file_path, dest_path)
                logger.info("File moved to processed/: %s", dest_path)
            else:
                logger.error(
                    "Ingestion FAILED for %s — file left in %s for inspection",
                    filename, SRC_DIR,
                )

        time.sleep(10)


if __name__ == "__main__":
    start_watcher()