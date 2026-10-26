"""
utils/timing_log.py
===================
Centralized structured timing-event logger for the NEAPS pipeline.

Example output:
  10:00:00.123  [TIMING] INGEST_START    elapsed_ms=0       request_id=a3f1  document_id=ANN_12345  worker_id=PID-4821
  10:00:04.300  [TIMING] QDRANT_END      elapsed_ms=4177    request_id=a3f1  document_id=ANN_12345  worker_id=PID-4821  chunks=84
  10:00:04.301  [TIMING] LLM_QUEUE       elapsed_ms=4178    request_id=a3f1  document_id=ANN_12345  worker_id=PID-4821  attempt=1
  10:00:04.305  [TIMING] LLM_START       elapsed_ms=4182    request_id=a3f1  document_id=ANN_12345  worker_id=PID-4821  attempt=1
  10:00:32.305  [TIMING] LLM_END         elapsed_ms=32182   request_id=a3f1  document_id=ANN_12345  worker_id=PID-4821  attempt=1  tokens=312
  10:00:32.310  [TIMING] INGEST_END      elapsed_ms=32187   request_id=a3f1  document_id=ANN_12345  worker_id=PID-4821  status=success
"""

import logging
import os
import time
import uuid
from typing import Any

_tlog = logging.getLogger("timing")
_EVENT_WIDTH = 14


def _fmt_event(event: str) -> str:
    return event.ljust(_EVENT_WIDTH)


class TimingLogger:
    """
    Lightweight timing logger tied to a single pipeline job (document / request).
    Thread-safe: all state is instance-local.

    Args:
        request_id:  Short unique ID for the top-level request or batch run.
        document_id: Identifier for the specific document or ANN_SEQ.
        worker_id:   Human-readable worker identifier, e.g. "PID-4821".
    """

    def __init__(
        self,
        request_id: str = "",
        document_id: str = "",
        worker_id: str = "",
    ) -> None:
        self.request_id = request_id or uuid.uuid4().hex[:8]
        self.document_id = document_id or ""
        self.worker_id = worker_id or f"PID-{os.getpid()}"
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Milliseconds since this TimingLogger was created."""
        return (time.perf_counter() - self._t0) * 1000

    def _build_fields(self, attempt: int, extra: dict) -> str:
        ms = self.elapsed_ms()
        fields = [f"elapsed_ms={ms:.0f}", f"request_id={self.request_id}"]
        if self.document_id:
            fields.append(f"document_id={self.document_id}")
        fields.append(f"worker_id={self.worker_id}")
        if attempt:
            fields.append(f"attempt={attempt}")
        for k, v in extra.items():
            fields.append(f"{k}={v}")
        return ms, "  ".join(fields)

    def emit(self, event: str, attempt: int = 0, **extra: Any) -> float:
        """Emit a structured timing log line at INFO level."""
        ms, field_str = self._build_fields(attempt, extra)
        _tlog.info("[TIMING] %s %s", _fmt_event(event), field_str)
        return ms

    def emit_error(self, event: str, error: str, attempt: int = 0, **extra: Any) -> float:
        """Emit a timing event at ERROR level (LLM_ERROR, INGEST_FAIL, etc.)."""
        ms, field_str = self._build_fields(attempt, extra)
        short_err = str(error)[:200].replace("\n", " ")
        _tlog.error("[TIMING] %s %s  error=%r", _fmt_event(event), field_str, short_err)
        return ms
