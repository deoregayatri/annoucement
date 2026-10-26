"""
Ingestion router - job monitoring (Excel/CSV endpoints disabled, using NEAPS only).

Endpoints:
  GET  /jobs/status             - real-time job queue statistics from Oracle
"""

import logging

from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger(__name__)

router = APIRouter()


# -----------------------------------------------------------------------
# Excel/CSV ingestion endpoints DISABLED — system uses NEAPS SFTP only.
# Uncomment below to re-enable JSON/Excel ingestion pipeline.
# -----------------------------------------------------------------------

# import os
# import tempfile
# from pathlib import Path
# from fastapi import UploadFile, File
# from config import settings
#
# @router.post("/ingest/excel-csv")
# async def ingest_excel_csv(file: UploadFile = File(...)):
#     """
#     Upload an Excel (.xlsx) or CSV (.csv) for JSON ingestion.
#
#     Required columns: company_name, company_symbol, title, platform, mentioned_date, url
#     Each valid row is fetched via URL and indexed into announcements_json.
#     """
#     filename = file.filename or "upload"
#     suffix   = Path(filename).suffix.lower()
#
#     if suffix not in (".xlsx", ".csv"):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Invalid file type '{suffix}'. Only .xlsx and .csv are accepted.",
#         )
#
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             tmp.write(await file.read())
#             tmp_path = Path(tmp.name)
#
#         try:
#             from ingestion.excel_csv_reader import ExcelCSVIngestionManager
#             mgr   = ExcelCSVIngestionManager()
#             stats = mgr.process_file(tmp_path)
#         finally:
#             try:
#                 os.unlink(tmp_path)
#             except Exception:
#                 pass
#
#         return {
#             "success"           : True,
#             "filename"          : filename,
#             "total_rows"        : stats.get("total", 0),
#             "dispatched"        : stats.get("dispatched", 0),
#             "skipped_duplicate" : stats.get("skipped_duplicate", 0),
#             "skipped_invalid"   : stats.get("skipped_invalid", 0),
#             "errors"            : stats.get("errors", []),
#         }
#
#     except HTTPException:
#         raise
#     except Exception as exc:
#         logger.error(f"[/ingest/excel-csv] Failed: {exc}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Ingestion failed: {str(exc)}",
#         )
#
#
# @router.post("/ingest/webexcel-folder")
# async def ingest_webexcel_folder(week_folder: str = ""):
#     """
#     Scan WEBEXCEL_DATA_DIR for Excel/CSV files and ingest them.
#
#     - Supply `week_folder` (e.g. '23-02-2026_to_27-02-2026') to process one sub-folder.
#     - Omit to process all sub-folders.
#     """
#     root_dir = Path(str(settings.webexcel_data_dir))
#
#     if not root_dir.exists():
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"WEBEXCEL_DATA_DIR not found: '{root_dir}'.",
#         )
#
#     if week_folder:
#         clean_folder = os.path.basename(week_folder.replace('\\', '/'))
#         if week_folder != clean_folder or ".." in week_folder:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid week_folder path."
#             )
#         target_dirs = [root_dir / clean_folder]
#     else:
#         target_dirs = [root_dir] + [p for p in root_dir.iterdir() if p.is_dir()]
#
#     all_files: list = []
#     for d in target_dirs:
#         if not d.exists():
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Week sub-folder not found: '{d}'",
#             )
#         all_files.extend(
#             f for f in d.iterdir()
#             if f.is_file() and f.suffix.lower() in (".xlsx", ".csv")
#         )
#
#     if not all_files:
#         return {
#             "success"        : True,
#             "message"        : "No .xlsx or .csv files found.",
#             "root_dir"       : str(root_dir),
#             "week_folder"    : week_folder or "(all)",
#             "files_processed": 0,
#             "results"        : [],
#         }
#
#     from ingestion.excel_csv_reader import ExcelCSVIngestionManager
#     mgr             = ExcelCSVIngestionManager()
#     results         = []
#     total_dispatched = 0
#
#     for filepath in sorted(all_files):
#         logger.info(f"[/ingest/webexcel-folder] Processing: {filepath.name}")
#         try:
#             s = mgr.process_file(filepath)
#             total_dispatched += s.get("dispatched", 0)
#             results.append({
#                 "file"             : filepath.name,
#                 "week_folder"      : filepath.parent.name,
#                 "status"           : "ok",
#                 "total_rows"       : s.get("total", 0),
#                 "dispatched"       : s.get("dispatched", 0),
#                 "skipped_duplicate": s.get("skipped_duplicate", 0),
#                 "skipped_invalid"  : s.get("skipped_invalid", 0),
#                 "errors"           : s.get("errors", []),
#             })
#         except Exception as exc:
#             logger.error(f"[/ingest/webexcel-folder] Failed for {filepath.name}: {exc}", exc_info=True)
#             results.append({
#                 "file"       : filepath.name,
#                 "week_folder": filepath.parent.name,
#                 "status"     : "error",
#                 "error"      : str(exc),
#             })
#
#     logger.info(
#         f"[/ingest/webexcel-folder] Done - {len(all_files)} files, {total_dispatched} jobs dispatched."
#     )
#     return {
#         "success"         : True,
#         "root_dir"        : str(root_dir),
#         "week_folder"     : week_folder or "(all)",
#         "files_processed" : len(all_files),
#         "total_dispatched": total_dispatched,
#         "results"         : results,
#     }


@router.get("/jobs/status", summary="Job Queue Status")
async def get_jobs_status():
    """
    Return real-time counts from the Oracle job_queue table.
    Returns available=false when in no-op mode (no Oracle DB).
    """
    try:
        from ingestion.job_store import get_job_store

        return get_job_store().get_stats()
    except Exception as exc:
        logger.error(f"[/jobs/status] Failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job status unavailable: {str(exc)}",
        )
