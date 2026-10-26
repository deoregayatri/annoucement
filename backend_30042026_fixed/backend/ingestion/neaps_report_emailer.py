"""
NEAPS Ingestion Report Generator & Email Sender.

- Generates a .txt report per symbol after run_ingestion() completes.
- Fetches full per-file error messages from AI_NEAPS_TRACKER (Oracle).
- Emails the report via Linux mailx (must be installed and configured on server).
- Auto-deletes reports older than NEAPS_REPORT_RETENTION_DAYS (default: 45 days).

Usage (called from workers/handlers.py after run_ingestion):
    from ingestion.neaps_report_emailer import generate_and_send_report
    generate_and_send_report(symbol, stats)
"""

import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_and_send_report(symbol: str, stats: Dict[str, Any]) -> None:
    """
    Main entry point called from the handler after run_ingestion() returns.

    Steps:
      1. Enrich stats with full error messages from AI_NEAPS_TRACKER.
      2. Generate .txt report file.
      3. Send via mailx.
      4. Cleanup old reports (>45 days).

    This function is fully non-fatal: any exception is logged and swallowed.
    """
    try:
        from config import settings
        report_dir       = getattr(settings, "neaps_report_dir",
                                   "/mnt/Announcement/backend/files_info")
        email_enabled    = getattr(settings, "neaps_report_email_enabled", True)
        recipients_str   = getattr(settings, "neaps_report_email_to", "")
        retention_days   = int(getattr(settings, "neaps_report_retention_days", 45))
    except Exception as cfg_exc:
        logger.warning(f"[Emailer] Could not load settings: {cfg_exc}. Using defaults.")
        report_dir       = "/mnt/Announcement/backend/files_info"
        email_enabled    = True
        recipients_str   = ""
        retention_days   = 45

    # Step 1: Enrich with Oracle error messages
    enriched_stats = _enrich_with_oracle_errors(symbol, stats)

    # Step 2: Generate report file
    report_path = _generate_report(symbol, enriched_stats, report_dir)
    if report_path is None:
        logger.error("[Emailer] Report generation failed. Skipping email.")
        return

    # Step 3: Send email if enabled and recipients configured
    if email_enabled and recipients_str.strip():
        recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
        _send_via_mailx(symbol, report_path, enriched_stats, recipients)
    else:
        logger.info("[Emailer] Email disabled or no recipients configured. Skipping email.")

    # Step 4: Cleanup old reports
    _cleanup_old_reports(report_dir, retention_days)


# ---------------------------------------------------------------------------
# Step 1: Enrich failed_files with Oracle error messages
# ---------------------------------------------------------------------------

def _enrich_with_oracle_errors(symbol: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query AI_NEAPS_TRACKER to get full ERROR_MESSAGE for failed files.
    Merges into stats['failed_files'] list.
    """
    import copy
    enriched = copy.deepcopy(stats)

    failed_files: List[Dict] = enriched.get("failed_files", [])
    if not failed_files:
        return enriched

    try:
        from ingestion.neaps_duplicate_checker import NeapsDuplicateChecker
        checker = NeapsDuplicateChecker()
        conn = checker._get_conn()
        if conn is None:
            logger.warning("[Emailer] Oracle not available. Using in-memory error messages.")
            return enriched

        # Collect all ann_seqs to query
        ann_seqs = [f["ann_seq"] for f in failed_files if f.get("ann_seq") not in ("?", None)]
        if not ann_seqs:
            return enriched

        placeholders = ", ".join(f":s{i}" for i in range(len(ann_seqs)))
        binds = {f"s{i}": seq for i, seq in enumerate(ann_seqs)}

        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT ANN_SEQ, ERROR_MESSAGE, ANN_FILE_ATTCH
            FROM   AI_NEAPS_TRACKER
            WHERE  ANN_SEQ IN ({placeholders})
            AND    PROCESS_STATUS IN ('FAILED', 'ERROR')
            """,
            binds,
        )
        rows = cur.fetchall()
        oracle_errors = {int(r[0]): {"error": r[1] or "Unknown error", "filename": r[2]}
                         for r in rows}
        cur.close()

        # Merge Oracle error messages into failed_files
        for f in failed_files:
            seq = f.get("ann_seq")
            if seq and seq != "?" and int(seq) in oracle_errors:
                f["error"]    = oracle_errors[int(seq)]["error"] or f.get("error", "")
                if not f.get("filename") or f["filename"] == "unknown":
                    f["filename"] = oracle_errors[int(seq)]["filename"] or f["filename"]

    except Exception as exc:
        logger.warning(f"[Emailer] Oracle error enrichment failed (non-fatal): {exc}")

    return enriched


# ---------------------------------------------------------------------------
# Step 2: Generate .txt report file
# ---------------------------------------------------------------------------

def _generate_report(symbol: str, stats: Dict[str, Any], report_dir: str) -> str | None:
    """
    Write a formatted .txt report to report_dir.
    Returns the absolute path to the created file, or None on failure.
    """
    try:
        Path(report_dir).mkdir(parents=True, exist_ok=True)

        now       = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        date_str  = now.strftime("%d-%b-%Y %H:%M")
        filename  = f"{symbol.upper()}_ingestion_report_{timestamp}.txt"
        filepath  = Path(report_dir) / filename

        successful_files: List[Dict] = stats.get("successful_files", [])
        failed_files:     List[Dict] = stats.get("failed_files", [])

        lines = [
            "=" * 60,
            f"  NEAPS Ingestion Report — {symbol.upper()} — {date_str}",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Total Announcements   : {stats.get('total', 0)}",
            f"Already Skipped       : {stats.get('skipped', 0)}",
            f"Downloaded            : {stats.get('downloaded', 0)}",
            f"Successfully Ingested : {stats.get('ingested', 0)}",
            f"Failed (Processing)   : {stats.get('failed_proc', 0)}",
            f"Failed (Download)     : {stats.get('failed_dl', 0)}",
            f"Skipped (Non-PDF)     : {stats.get('skipped_type', 0)}",
            f"Total Chunks Created  : {stats.get('total_chunks', 0)}",
            "",
        ]

        # Successfully processed files
        lines += ["=" * 60, "SUCCESSFULLY PROCESSED FILES", "=" * 60]
        if successful_files:
            for i, f in enumerate(successful_files, 1):
                lines.append(
                    f"{i}. {f.get('filename', 'unknown')}"
                    f"  (ann_seq={f.get('ann_seq','?')}, chunks={f.get('chunks',0)})"
                )
        else:
            lines.append("  None")
        lines.append("")

        # Failed files
        lines += ["=" * 60, "FAILED FILES", "=" * 60]
        if failed_files:
            for i, f in enumerate(failed_files, 1):
                lines.append(f"{i}. {f.get('filename', 'unknown')}")
                lines.append(f"   ann_seq : {f.get('ann_seq', '?')}")
                lines.append(f"   Error   : {f.get('error', 'Unknown error')}")
                lines.append("")
        else:
            lines.append("  None")
        lines.append("")

        lines += [
            "=" * 60,
            f"Report generated at: {date_str}",
            "This is an automated report from NEAPS Ingestion System.",
            "=" * 60,
        ]

        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        logger.info(f"[Emailer] Report saved: {filepath}")
        return str(filepath)

    except Exception as exc:
        logger.error(f"[Emailer] Failed to generate report: {exc}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Step 3: Send via mailx
# ---------------------------------------------------------------------------

def _send_via_mailx(
    symbol: str,
    report_path: str,
    stats: Dict[str, Any],
    recipients: List[str],
) -> None:
    """
    Send the report file as an email attachment using Linux mailx.
    mailx must be installed and configured on the server (/etc/mail.rc).
    Non-fatal: logs warning on failure.
    """
    if not recipients:
        logger.warning("[Emailer] No recipients configured. Skipping email.")
        return

    now      = datetime.now()
    date_str = now.strftime("%d-%b-%Y")
    subject  = f"Ingestion Report — {symbol.upper()} — {date_str}"

    body = (
        f"NEAPS Ingestion Report — {symbol.upper()} — {date_str}\n\n"
        f"Summary:\n"
        f"  Total Announcements   : {stats.get('total', 0)}\n"
        f"  Already Skipped       : {stats.get('skipped', 0)}\n"
        f"  Successfully Ingested : {stats.get('ingested', 0)}\n"
        f"  Failed (Processing)   : {stats.get('failed_proc', 0)}\n"
        f"  Failed (Download)     : {stats.get('failed_dl', 0)}\n"
        f"  Total Chunks Created  : {stats.get('total_chunks', 0)}\n\n"
        f"Full report is attached.\n\n"
        f"This is an automated message from the NEAPS Ingestion System."
    )

    # mailx command: echo body | mailx -s "subject" -a attachment recipient1 recipient2
    cmd = [
        "mailx",
        "-s", subject,
        "-a", report_path,
    ] + recipients

    try:
        result = subprocess.run(
            cmd,
            input=body,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info(
                f"[Emailer] Email sent to {recipients} for symbol={symbol}"
            )
        else:
            logger.warning(
                f"[Emailer] mailx returned non-zero ({result.returncode}): "
                f"{result.stderr.strip()}"
            )
    except FileNotFoundError:
        logger.warning(
            "[Emailer] mailx not found on server. "
            "Ask Linux team to install: sudo yum install mailx"
        )
    except subprocess.TimeoutExpired:
        logger.warning("[Emailer] mailx timed out after 30s.")
    except Exception as exc:
        logger.warning(f"[Emailer] Email send failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Step 4: Cleanup old reports
# ---------------------------------------------------------------------------

def _cleanup_old_reports(report_dir: str, retention_days: int = 45) -> None:
    """
    Delete .txt report files older than retention_days from report_dir.
    """
    try:
        cutoff = datetime.now() - timedelta(days=retention_days)
        report_path = Path(report_dir)
        if not report_path.exists():
            return

        deleted = 0
        for txt_file in report_path.glob("*_ingestion_report_*.txt"):
            mtime = datetime.fromtimestamp(txt_file.stat().st_mtime)
            if mtime < cutoff:
                txt_file.unlink()
                deleted += 1
                logger.debug(f"[Emailer] Deleted old report: {txt_file.name}")

        if deleted:
            logger.info(
                f"[Emailer] Cleaned up {deleted} report(s) older than {retention_days} days."
            )
    except Exception as exc:
        logger.warning(f"[Emailer] Cleanup failed (non-fatal): {exc}")
