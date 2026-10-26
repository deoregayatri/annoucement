# -*- coding: utf-8 -*-
"""
run_neaps_ingest.py
===================
Standalone CLI script -- run the full NEAPS ingestion pipeline.

Pipeline flow for EACH company:
  Oracle (metadata) -> duplicate-check -> SFTP download -> text extract
  -> chunk -> enrich -> hybrid embed -> Qdrant upsert -> mark COMPLETED

Usage:
  # Ingest one company
  python run_neaps_ingest.py --company TCS

  # Ingest multiple companies
  python run_neaps_ingest.py --company TCS INFY RELIANCE

  # Ingest ALL companies found in Oracle NEAPSDBA
  python run_neaps_ingest.py --all

  # Dry-run: show what WOULD be ingested without actually doing it
  python run_neaps_ingest.py --company TCS --dry-run

  # Force re-ingest even if already marked COMPLETED
  python run_neaps_ingest.py --company TCS --force

  # Limit announcements per company (useful for first test)
  python run_neaps_ingest.py --company TCS --limit 10

All credentials and paths are read from .env -- nothing is hardcoded.
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from typing import List

# -- Logging setup ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neaps_ingest")


# -- Helpers ------------------------------------------------------------------

def _print_banner():
    print("\n" + "=" * 60)
    print("  NEAPS INGESTION PIPELINE")
    print("  Started : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")


def _print_stats(symbol, stats):
    print("\n  -- " + symbol + " --")
    print("     Total announcements  : " + str(stats.get("total", 0)))
    print("     Already processed    : " + str(stats.get("skipped", 0)))
    print("     SFTP downloaded      : " + str(stats.get("downloaded", 0)))
    print("     SFTP failed          : " + str(stats.get("failed_dl", 0)))
    print("     Ingested to Qdrant   : " + str(stats.get("ingested", 0)))
    print("     Processing failures  : " + str(stats.get("failed_proc", 0)))
    print("     Total chunks stored  : " + str(stats.get("total_chunks", 0)))


def _get_all_symbols():
    """Fetch every company symbol from Oracle NEAPSDBA."""
    try:
        from ingestion.neaps_oracle_client import get_neaps_oracle_client
        symbols = get_neaps_oracle_client().get_unique_symbols()
        logger.info("Oracle returned %d companies", len(symbols))
        return symbols
    except Exception as exc:
        logger.error("Could not fetch symbols from Oracle: %s", exc)
        sys.exit(1)


def _reset_tracking(symbols):
    """
    Delete AI_NEAPS_PROCESSED records for the given symbol list so they
    get re-processed even if previously marked COMPLETED.
    """
    from ingestion.neaps_oracle_client import get_neaps_oracle_client
    from ingestion.neaps_duplicate_checker import get_neaps_duplicate_checker

    oracle  = get_neaps_oracle_client()
    checker = get_neaps_duplicate_checker()
    conn    = checker._get_conn()
    if conn is None:
        logger.warning("[force] Cannot connect to Oracle -- skipping reset.")
        return

    for sym in symbols:
        anns = oracle.fetch_announcements(sym)
        seqs = [int(a.get("ann_seq", 0)) for a in anns if a.get("ann_seq")]
        if not seqs:
            continue
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM AI_NEAPS_PROCESSED WHERE ANN_SEQ IN (%s)"
            % ",".join(str(s) for s in seqs)
        )
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        logger.info("[force] Reset %d record(s) for '%s'", deleted, sym)


def _dry_run(symbol, limit):
    """Show what announcements exist without ingesting anything."""
    from ingestion.neaps_oracle_client import get_neaps_oracle_client
    from ingestion.neaps_duplicate_checker import get_neaps_duplicate_checker

    oracle  = get_neaps_oracle_client()
    checker = get_neaps_duplicate_checker()

    anns = oracle.fetch_announcements(symbol)
    if limit:
        anns = anns[:limit]

    pending = [a for a in anns if not checker.is_processed(int(a.get("ann_seq", 0)))]

    print("\n  [DRY-RUN] " + symbol)
    print("     Total in Oracle      : " + str(len(anns)))
    print("     Already processed    : " + str(len(anns) - len(pending)))
    print("     Would ingest         : " + str(len(pending)))
    if pending:
        print("     Pending ANN_SEQs:")
        for a in pending[:5]:
            print("       seq=" + str(a.get("ann_seq")) +
                  "  file=" + str(a.get("ann_file_attach", "N/A")))
        if len(pending) > 5:
            print("       ... and " + str(len(pending) - 5) + " more")
    print()


# -- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="NEAPS ingestion pipeline -- fetch from Oracle+SFTP -> Qdrant"
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--company", nargs="+", metavar="SYMBOL",
        help="One or more company symbols (e.g. TCS INFY)",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Ingest ALL companies found in Oracle NEAPSDBA",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be ingested -- no SFTP download, no Qdrant writes",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Delete existing COMPLETED records and re-ingest (use after deleting local files)",
    )
    ap.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Process at most N announcements per company (0 = unlimited)",
    )
    args = ap.parse_args()

    _print_banner()

    # Resolve symbols list
    symbols = [s.upper() for s in args.company] if args.company else _get_all_symbols()

    if not symbols:
        logger.error("No company symbols to process -- exiting.")
        sys.exit(1)

    logger.info("Companies to process: %s", symbols)

    # Dry-run path
    if args.dry_run:
        print("\n[DRY-RUN MODE -- no data will be written]\n")
        for sym in symbols:
            _dry_run(sym, args.limit)
        print("Dry-run complete.\n")
        return

    # Real ingestion
    from ingestion.neaps_ingestion_service import run_ingestion

    # Force-reset tracking records if requested
    if args.force:
        logger.info("[force] Resetting tracking records for: %s", symbols)
        _reset_tracking(symbols)
    grand = {
        "total": 0, "skipped": 0, "downloaded": 0,
        "failed_dl": 0, "ingested": 0, "failed_proc": 0, "total_chunks": 0
    }
    t0 = time.time()

    for i, sym in enumerate(symbols, 1):
        logger.info("[%d/%d] Starting ingestion for '%s' ...", i, len(symbols), sym)
        try:
            stats = run_ingestion(sym)
            _print_stats(sym, stats)
            for k in grand:
                grand[k] += stats.get(k, 0)
        except Exception as exc:
            logger.error("Pipeline failed for '%s': %s", sym, exc, exc_info=True)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  GRAND TOTAL")
    _print_stats("ALL", grand)
    print("\n  Elapsed : %.1fs" % elapsed)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
 