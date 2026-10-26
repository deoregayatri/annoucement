"""
Structured JSON Logger - Stage 9.

Writes every event as a single-line JSON object to logs/structured.jsonl.
Human-readable format is kept for console output.

Every log line contains:
    timestamp, query_id, company_id, stage, latency_ms
    + any additional kwargs

Usage:
    from utils.structured_logger import get_structured_logger
    slog = get_structured_logger()
    slog.log_query_event(query_id="abc", stage="retrieval", latency_ms=123.4, json_chunks=8)
    slog.log_ingestion_event(doc_id="xyz", stage="embed", status="SUCCESS")
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

_STRUCTURED_LOG_FILE = Path(settings.log_dir) / "structured.jsonl"


class StructuredLogger:
    """
    Writes JSON Lines to structured.jsonl for machine consumption.
    Falls back gracefully if the log directory isn't writable.
    """

    def __init__(self) -> None:
        try:
            _STRUCTURED_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(_STRUCTURED_LOG_FILE, "a", encoding="utf-8", buffering=1)
            logger.info(f"[StructuredLogger] Logging to {_STRUCTURED_LOG_FILE}")
        except Exception as exc:
            logger.warning(f"[StructuredLogger] Cannot open log file: {exc}. Events will be skipped.")
            self._fh = None

    def _write(self, record: dict) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.debug(f"[StructuredLogger] Write failed: {exc}")

    @staticmethod
    def _base(event_type: str) -> dict:
        return {
            "timestamp"  : datetime.now(timezone.utc).isoformat(),
            "event_type" : event_type,
        }

    def log_query_event(
        self,
        query_id: Optional[str] = None,
        company_id: Optional[str] = None,
        stage: str = "unknown",
        latency_ms: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Log a query pipeline stage event."""
        record = self._base("query")
        record.update({
            "query_id"   : query_id or str(uuid.uuid4()),
            "company_id" : company_id,
            "stage"      : stage,
            "latency_ms" : round(latency_ms, 2),
        })
        record.update(kwargs)
        self._write(record)

    def log_ingestion_event(
        self,
        doc_id: Optional[str] = None,
        stage: str = "unknown",
        status: str = "unknown",
        **kwargs: Any,
    ) -> None:
        """Log a document ingestion stage event."""
        record = self._base("ingestion")
        record.update({
            "doc_id": doc_id,
            "stage" : stage,
            "status": status,
        })
        record.update(kwargs)
        self._write(record)

    def log_celery_event(
        self,
        task_name: str,
        row_id: Optional[str] = None,
        status: str = "unknown",
        **kwargs: Any,
    ) -> None:
        """Log a Celery task event."""
        record = self._base("celery")
        record.update({
            "task_name": task_name,
            "row_id"   : row_id,
            "status"   : status,
        })
        record.update(kwargs)
        self._write(record)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_structured_logger: Optional[StructuredLogger] = None


def get_structured_logger() -> StructuredLogger:
    """Get or create the StructuredLogger singleton."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger()
    return _structured_logger
