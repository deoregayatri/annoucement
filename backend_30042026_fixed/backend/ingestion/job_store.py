"""
Job Store -- Oracle DB-backed job queue.

Replaces Redis + Celery broker with a DB-native job queue.

Table: job_queue
  job_id         VARCHAR2(36) PK
  job_type       VARCHAR2(100)  -- 'json_fetch', 'excel_ingestion', etc.
  payload        CLOB           -- JSON-serialised payload
  status         VARCHAR2(20)   -- pending | processing | completed | failed
  retry_count    NUMBER(5)
  max_retries    NUMBER(5)
  error_message  CLOB
  created_at     TIMESTAMP
  updated_at     TIMESTAMP
  locked_by      VARCHAR2(100)  -- worker_id that claimed the job
  locked_at      TIMESTAMP      -- when it was claimed

Concurrency:
  claim_jobs() uses SELECT FOR UPDATE SKIP LOCKED ensuring no two workers
  pick the same row, with zero Redis/Celery dependency.

Fallback:
  If Oracle is not configured, JobStore runs in no-op mode and callers must
  handle jobs synchronously (same behaviour as the old Celery stub).
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Allowlist of valid job types -- reject anything else at enqueue time
VALID_JOB_TYPES = frozenset({
    "json_fetch",
    "excel_ingestion",
    "document_ingestion",
    "dictionary_extraction",
    "vlm_image_processing",
    "neaps_ingestion",          # NEAPS symbol ingestion via UI
    "neaps_ingestion_batch",    # Sub-batch of PDFs for interleaved processing
})

# Maximum serialised payload size (bytes). Protects Oracle CLOB + temp segment.
MAX_PAYLOAD_BYTES = 512 * 1024   # 512 KB

# Default pool connection-acquire timeout (seconds). Prevents deadlock on pool exhaustion.
POOL_ACQUIRE_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_JOB_QUEUE = """
CREATE TABLE job_queue (
    job_id          VARCHAR2(36)    PRIMARY KEY,
    job_type        VARCHAR2(100)   NOT NULL,
    symbol          VARCHAR2(100),
    payload         CLOB            NOT NULL,
    status          VARCHAR2(20)    DEFAULT 'pending' NOT NULL,
    retry_count     NUMBER(5)       DEFAULT 0 NOT NULL,
    max_retries     NUMBER(5)       DEFAULT 3 NOT NULL,
    retry_after     TIMESTAMP,
    error_message   CLOB,
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP NOT NULL,
    locked_by       VARCHAR2(100),
    locked_at       TIMESTAMP
)
"""

_IDX_JOB_STATUS   = "CREATE INDEX idx_jq_status   ON job_queue(status)"
_IDX_JOB_TYPE     = "CREATE INDEX idx_jq_type     ON job_queue(job_type)"
_IDX_JOB_LOCKED   = "CREATE INDEX idx_jq_locked   ON job_queue(locked_at)"
# Composite index for the hot claim_jobs query: WHERE status='pending' ORDER BY created_at
_IDX_JOB_CLAIM    = "CREATE INDEX idx_jq_claim    ON job_queue(status, created_at)"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

class Job:
    """Lightweight job record returned by claim_jobs()."""

    __slots__ = ("job_id", "job_type", "payload", "retry_count", "max_retries", "created_at")

    def __init__(self, job_id: str, job_type: str, payload: Dict[str, Any],
                 retry_count: int, max_retries: int, created_at=None) -> None:
        self.job_id      = job_id
        self.job_type    = job_type
        self.payload     = payload
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.created_at  = created_at   # datetime � used to compute queue_wait_ms

    def __repr__(self) -> str:
        return f"<Job id={self.job_id[:8]} type={self.job_type} retry={self.retry_count}>"


# ---------------------------------------------------------------------------
# JobStore
# ---------------------------------------------------------------------------

class JobStore:
    """
    Wraps all job_queue table operations.

    Thread-safe: each public method acquires a fresh connection from the
    Oracle pool (or a single connection in fallback mode).
    """

    def __init__(self) -> None:
        self._pool: Any   = None
        self._conn: Any   = None       # fallback single connection
        self._available   = None       # None = untested
        self._init_lock   = threading.Lock()    # guards _connect() TOCTOU
        logger.info("[JobStore] Initialised (lazy Oracle connect).")

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Establish Oracle connection pool. Thread-safe via double-checked Lock."""
        if self._available is not None:     # fast path (no lock needed after init)
            return self._available

        with self._init_lock:               # only one thread initialises the pool
            if self._available is not None: # re-check inside lock
                return self._available

            try:
                import oracledb
                from config import settings

                user = getattr(settings, "oracle_user",     None) or ""
                pwd  = getattr(settings, "oracle_password", None) or ""
                dsn  = getattr(settings, "oracle_dsn",      None) or ""

                missing = [n for n, v in (("oracle_user", user), ("oracle_password", pwd), ("oracle_dsn", dsn)) if not v]
                if missing:
                    logger.warning(f"[JobStore] Missing Oracle config keys: {missing} -- no-op mode.")
                    self._available = False
                    return False

                # ----------------------------------------------------------
                # Thick mode: required for NNE (Native Network Encryption),
                # which is the default on corporate Oracle servers.
                # ----------------------------------------------------------
                thick_mode     = getattr(settings, "oracle_thick_mode",      False)
                client_lib_dir = getattr(settings, "oracle_client_lib_dir",  "") or ""
                if thick_mode:
                    try:
                        init_kwargs = {"lib_dir": client_lib_dir} if client_lib_dir else {}
                        oracledb.init_oracle_client(**init_kwargs)
                        logger.info("[JobStore] oracledb thick mode enabled.")
                    except Exception as thick_exc:
                        # init_oracle_client raises if called twice -- safe to ignore
                        if "already been called" not in str(thick_exc).lower():
                            logger.warning(f"[JobStore] Thick mode init warning: {thick_exc}")

                self._pool = oracledb.create_pool(
                    user=user, password=pwd, dsn=dsn,
                    min=2, max=10, increment=1,
                    getmode=oracledb.POOL_GETMODE_TIMEDWAIT,
                    wait_timeout=POOL_ACQUIRE_TIMEOUT_S * 1000,  # milliseconds
                )
                self._available = True
                logger.info("[JobStore] Oracle connection pool established.")
                self._ensure_table()
                return True

            except Exception as exc:
                logger.warning(f"[JobStore] Oracle unavailable ({exc}) -- no-op mode.")
                self._available = False
                return False

    def _get_conn(self):
        """Acquire a connection from the pool.
        Timeout is already enforced at pool level via wait_timeout + POOL_GETMODE_TIMEDWAIT."""
        return self._pool.acquire()

    def _ensure_table(self) -> None:
        """Create job_queue table + indexes if they don't already exist."""
        stmts = [
            (_DDL_JOB_QUEUE,   "job_queue"),
            (_IDX_JOB_STATUS,  "idx_jq_status"),
            (_IDX_JOB_TYPE,    "idx_jq_type"),
            (_IDX_JOB_LOCKED,  "idx_jq_locked"),
            (_IDX_JOB_CLAIM,   "idx_jq_claim"),   # composite for hot claim_jobs query
        ]
        with self._get_conn() as conn:
            for ddl, obj_name in stmts:
                try:
                    conn.cursor().execute(ddl)
                    conn.commit()
                    logger.info(f"[JobStore] Created DB object: {obj_name}")
                except Exception as exc:
                    exc_str = str(exc)
                    # ORA-00955 = name already used; ORA-00942 = table/view does not exist
                    if "ORA-00955" in exc_str or "already exists" in exc_str.lower():
                        pass  # Already created -- fine
                    else:
                        logger.warning(f"[JobStore] DDL warning for {obj_name}: {exc_str}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Insert a new pending job into job_queue.

        Raises:
            ValueError: if job_type is not in VALID_JOB_TYPES, or payload is too large.

        Returns:
            job_id (UUID str) on success, None in no-op mode.
        """
        # -- Validation ----------------------------------------------------
        if job_type not in VALID_JOB_TYPES:
            raise ValueError(
                f"[JobStore] Unknown job_type={job_type!r}. "
                f"Valid types: {sorted(VALID_JOB_TYPES)}"
            )

        payload_json = json.dumps(payload, default=str)
        if len(payload_json.encode()) > MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"[JobStore] Payload too large ({len(payload_json.encode())} bytes > "
                f"{MAX_PAYLOAD_BYTES} bytes limit) for job_type={job_type!r}"
            )
        # -----------------------------------------------------------------

        if not self._connect():
            return None

        if job_type == "neaps_ingestion":
            symbol = payload.get("symbol", "")
            if symbol and self._is_symbol_active(symbol):
                logger.warning(
                    f"[JobStore] Skipping duplicate enqueue: "
                    f"symbol={symbol} already has active job"
                )
                return None

        job_id = str(uuid.uuid4())
        now    = datetime.now(timezone.utc)

        try:
            with self._get_conn() as conn:
                conn.cursor().execute(
                    """
                    INSERT INTO job_queue
                        (job_id, job_type, symbol, payload, status, retry_count, max_retries, created_at, updated_at)
                    VALUES
                        (:1, :2, :3, :4, 'pending', 0, :5, :6, :6)
                    """,
                    [job_id, job_type, payload.get("symbol"), payload_json, max_retries, now],
                )
                conn.commit()
            logger.info(f"[JobStore] Enqueued job_id={job_id[:8]} type={job_type}")
            return job_id
        except Exception as exc:
            logger.error(f"[JobStore] insert_job failed: {exc}", exc_info=True)
            return None

    def _is_symbol_active(self, symbol: str) -> bool:
        """True if symbol already has pending/processing neaps_ingestion job."""
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT COUNT(*) FROM job_queue
                    WHERE  job_type = 'neaps_ingestion'
                    AND    status   IN ('pending', 'processing')
                    AND    symbol = :sym
                    """,
                    {"sym": symbol.upper()}
                )
                return cur.fetchone()[0] > 0
        except Exception as exc:
            logger.error(f"[JobStore] _is_symbol_active failed: {exc}", exc_info=True)
            return False

    def claim_jobs(self, n: int, worker_id: str) -> List[Job]:
        """
        Atomically claim up to N pending jobs using SELECT FOR UPDATE SKIP LOCKED.
        Marks them as 'processing' in the same transaction.

        Args:
            n:         Max jobs to claim
            worker_id: Unique identifier for this worker thread/process

        Returns:
            List of Job objects (may be empty if nothing pending).
        """
        if not self._connect():
            return []

        try:
            now = datetime.now(timezone.utc)
            jobs: List[Job] = []

            with self._get_conn() as conn:
                cur = conn.cursor()

                # ---------------------------------------------------------
                # Oracle ORA-02014 workaround:
                # FOR UPDATE cannot be used on a query containing
                # FETCH FIRST N ROWS ONLY (Oracle internally wraps it in a
                # ROWNUM view, making it un-lockable in one shot).
                #
                # Solution: two-step ROWID locking --
                #   Step 1: collect ROWIDs of candidate rows (ORDER BY + FETCH FIRST).
                #   Step 2: lock those rows by ROWID with FOR UPDATE SKIP LOCKED.
                # ---------------------------------------------------------

                # ----------------------------------------------------------
                # Global concurrency cap (DB-level, works across all server
                # instances): count currently processing neaps_ingestion jobs
                # and reduce n so we never exceed MAX_WORKERS_CPU globally.
                # ----------------------------------------------------------
                try:
                    from config import settings as _settings
                    _cpu_cap = max(1, int(getattr(_settings, "max_workers_cpu", 2)))
                except Exception:
                    _cpu_cap = 2

                cur.execute(
                    """
                    SELECT COUNT(*) FROM job_queue
                    WHERE  status   = 'processing'
                    AND    job_type = 'neaps_ingestion'
                    """
                )
                _active_neaps = cur.fetchone()[0]
                _neaps_slots  = max(0, _cpu_cap - _active_neaps)

                # Adjust n: for neaps_ingestion jobs, cap to available slots;
                # for other job types we do not restrict here (IO pool handles them).
                # We rely on the caller (worker_manager._available_slots) to pass
                # the right n, but this is the last safety net at DB level.
                if _neaps_slots == 0:
                    # No CPU slots left globally — skip claim entirely this poll
                    logger.debug(
                        f"[JobStore] Global neaps_ingestion cap reached "
                        f"(active={_active_neaps}/{_cpu_cap}). Skipping claim."
                    )
                    return []

                # Clamp n to remaining neaps slots so we never over-claim
                n = min(n, _neaps_slots)

                # Step 1: identify candidate row IDs (no FOR UPDATE here)
                cur.execute(
                    """
                    SELECT job_id
                    FROM   job_queue
                    WHERE  status = 'pending'
                    AND   (retry_after IS NULL OR retry_after <= :now)
                    ORDER  BY created_at ASC
                    FETCH  FIRST :n ROWS ONLY
                    """,
                    {"n": n, "now": now},
                )
                candidate_ids = [r[0] for r in cur.fetchall()]

                if not candidate_ids:
                    return []

                # Step 2: lock + read the selected rows (FOR UPDATE SKIP LOCKED)
                # Build bind variables :id0, :id1, ... dynamically
                placeholders = ", ".join(f":id{i}" for i in range(len(candidate_ids)))
                binds = {f"id{i}": jid for i, jid in enumerate(candidate_ids)}
                cur.execute(
                    f"""
                    SELECT job_id, job_type, payload, retry_count, max_retries, created_at
                    FROM   job_queue
                    WHERE  job_id IN ({placeholders})
                    FOR UPDATE SKIP LOCKED
                    """,
                    binds,
                )
                rows = cur.fetchall()

                for row in rows:
                    job_id_r, job_type_r, payload_str, retry_r, max_r, created_r = (
                        row[0], row[1], row[2], int(row[3]), int(row[4]),
                        row[5] if len(row) > 5 else None,
                    )
                    try:
                        # Oracle returns CLOB columns as a CLOB object, not a plain str.
                        # Must call .read() to materialise it before passing to json.loads.
                        if hasattr(payload_str, "read"):
                            payload_str = payload_str.read()
                        payload = json.loads(payload_str)
                    except Exception as _je:
                        logger.error(
                            f"[JobStore] Failed to deserialise payload for job_id={job_id_r}: {_je} "
                            f"(raw type={type(payload_str).__name__}, val={str(payload_str)[:100]})"
                        )
                        payload = {}

                    # Queue wait time � from job creation to now being claimed
                    wait_ms = 0.0
                    if created_r is not None:
                        try:
                            ts = created_r.replace(tzinfo=timezone.utc) if created_r.tzinfo is None else created_r
                            wait_ms = (now - ts).total_seconds() * 1000
                        except Exception:
                            pass

                    logger.info(
                        f"[JobStore] Claimed job_id={job_id_r[:8]} type={job_type_r} "
                        f"retry={retry_r} queue_wait={wait_ms:.0f}ms"
                    )

                    cur.execute(
                        """
                        UPDATE job_queue
                        SET    status    = 'processing',
                               locked_by = :worker_id,
                               locked_at = :now,
                               updated_at = :now
                        WHERE  job_id = :job_id
                        """,
                        {"worker_id": worker_id, "now": now, "job_id": job_id_r},
                    )
                    jobs.append(Job(
                        job_id      = job_id_r,
                        job_type    = job_type_r,
                        payload     = payload,
                        retry_count = retry_r,
                        max_retries = max_r,
                        created_at  = created_r,
                    ))

                conn.commit()

            if jobs:
                logger.debug(f"[JobStore] Claimed {len(jobs)} job(s) for worker={worker_id}")
            return jobs

        except Exception as exc:
            logger.error(f"[JobStore] claim_jobs failed: {exc}", exc_info=True)
            return []


    def mark_completed(self, job_id: str) -> None:
        """Mark a job as completed."""
        self._update_status(job_id, "completed")
        logger.info(f"[JobStore] job_id={job_id[:8]} -> completed")

    def mark_failed(self, job_id: str, error: str, retry_count: int, max_retries: int) -> None:
        """
        On failure:
          - If retry_count < max_retries  -> status = 'pending' with exponential back-off
          - Else                          -> status = 'failed'  (terminal)

        Back-off: min(2^retry_count * 30s, 3600s)
          retry 0 ->  30s,  retry 1 ->  60s,  retry 2 -> 120s ... max 1h
        """
        if not self._connect():
            return

        new_status  = "pending" if retry_count < max_retries else "failed"
        now         = datetime.now(timezone.utc)
        new_retry   = retry_count + 1

        # Compute retry_after only when re-queuing, to prevent hot-retry storms
        retry_after = None
        if new_status == "pending":
            from datetime import timedelta
            delay_s     = min(30 * (2 ** retry_count), 3600)   # 30s, 60s, 120s ... 3600s
            retry_after = now + timedelta(seconds=delay_s)
            logger.info(
                f"[JobStore] job_id={job_id[:8]} -> retry {new_retry}/{max_retries} "
                f"after {delay_s}s (at {retry_after.isoformat()})"
            )
        else:
            logger.warning(f"[JobStore] job_id={job_id[:8]} -> FAILED (exhausted {max_retries} retries)")

        try:
            with self._get_conn() as conn:
                conn.cursor().execute(
                    """
                    UPDATE job_queue
                    SET    status        = :status,
                           retry_count   = :retry_count,
                           retry_after   = :retry_after,
                           error_message = :error,
                           locked_by     = NULL,
                           locked_at     = NULL,
                           updated_at    = :now
                    WHERE  job_id = :job_id
                    """,
                    {
                        "status":      new_status,
                        "retry_count": new_retry,
                        "retry_after": retry_after,
                        "error":       error[:4000],
                        "now":         now,
                        "job_id":      job_id,
                    },
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[JobStore] mark_failed error: {exc}", exc_info=True)

    def heartbeat(self, job_id: str, worker_id: str) -> bool:
        """Refresh locked_at for a running job. Returns False if job was stolen."""
        if not self._connect():
            return False
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE job_queue
                    SET    locked_at  = :now,
                           updated_at = :now
                    WHERE  job_id     = :job_id
                    AND    locked_by  = :worker_id
                    AND    status     = 'processing'
                    """,
                    {
                        "now": datetime.now(timezone.utc),
                        "job_id": job_id,
                        "worker_id": worker_id,
                    },
                )
                conn.commit()
                return cur.rowcount == 1
        except Exception as exc:
            logger.error(f"[JobStore] heartbeat failed: {exc}", exc_info=True)
            return False

    def requeue_stale_jobs(self, lock_timeout_s: int = 1800) -> int:
        """
        Reset jobs that have been 'processing' without a heartbeat for lock_timeout_s seconds.
        This handles worker crashes.

        Returns:
            Number of jobs requeued.
        """
        if not self._connect():
            return 0

        try:
            now = datetime.now(timezone.utc)
            with self._get_conn() as conn:
                cur = conn.cursor()
                # Oracle: INTERVAL with a bind variable cannot be used inside the
                # literal string. Use NUMTODSINTERVAL() which accepts a numeric bind.
                cur.execute(
                    """
                    UPDATE job_queue
                    SET    status     = 'pending',
                           locked_by  = NULL,
                           locked_at  = NULL,
                           updated_at = :now
                    WHERE  status    = 'processing'
                    AND    locked_at < (:now - NUMTODSINTERVAL(:timeout, 'SECOND'))
                    """,
                    {"now": now, "timeout": lock_timeout_s},
                )
                count = cur.rowcount
                conn.commit()
            if count:
                logger.info(f"[JobStore] Requeued {count} stale job(s) (timeout={lock_timeout_s}s)")
            return count
        except Exception as exc:
            logger.error(f"[JobStore] requeue_stale_jobs failed: {exc}", exc_info=True)
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Return queue statistics for the /jobs/status monitoring endpoint.

        Returns dict with:
          total, pending, processing, completed, failed,
          by_type: {job_type: {status: count}}
        """
        if not self._connect():
            return {
                "available": False,
                "total": 0, "pending": 0, "processing": 0,
                "completed": 0, "failed": 0, "by_type": {},
            }

        try:
            with self._get_conn() as conn:
                cur = conn.cursor()

                # Overall counts
                cur.execute(
                    """
                    SELECT status, COUNT(*) FROM job_queue GROUP BY status
                    """
                )
                counts = {row[0]: int(row[1]) for row in cur.fetchall()}

                # Per-type breakdown
                cur.execute(
                    """
                    SELECT job_type, status, COUNT(*)
                    FROM   job_queue
                    GROUP  BY job_type, status
                    ORDER  BY job_type, status
                    """
                )
                by_type: Dict[str, Dict[str, int]] = {}
                for job_type, status, cnt in cur.fetchall():
                    by_type.setdefault(job_type, {})[status] = int(cnt)

            return {
                "available":  True,
                "total":      sum(counts.values()),
                "pending":    counts.get("pending",    0),
                "processing": counts.get("processing", 0),
                "completed":  counts.get("completed",  0),
                "failed":     counts.get("failed",     0),
                "by_type":    by_type,
            }
        except Exception as exc:
            logger.error(f"[JobStore] get_stats failed: {exc}", exc_info=True)
            return {"available": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_status(self, job_id: str, status: str) -> None:
        """Set job status and clear lock fields."""
        if not self._connect():
            return
        try:
            now = datetime.now(timezone.utc)
            with self._get_conn() as conn:
                conn.cursor().execute(
                    """
                    UPDATE job_queue
                    SET    status    = :status,
                           locked_by = NULL,
                           locked_at = NULL,
                           updated_at = :now
                    WHERE  job_id = :job_id
                    """,
                    {"status": status, "now": now, "job_id": job_id},
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[JobStore] _update_status({status}) failed: {exc}", exc_info=True)


    def get_pending_count(self) -> int:
        """
        Fast COUNT of pending jobs used by BackpressureController in worker_manager.py.

        Returns:
            Number of rows with status='pending'. Returns 0 on any DB error
            so the poll loop degrades gracefully rather than crashing.
        """
        if not self._connect():
            return 0
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM job_queue WHERE status = 'pending'"
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:
            logger.debug(f"[JobStore] get_pending_count failed: {exc}")
            return 0

    def get_active_neaps_symbols(self) -> Dict[str, str]:
        """
        Returns a dict of symbols to their status for active 'neaps_ingestion' jobs.
        Used by the API so the UI can show symbols that are in the queue waiting to start.
        e.g., {'SKYGOLD': 'processing', 'TCS': 'pending'}
        """
        if not self._connect():
            return {}
        try:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT symbol, status 
                    FROM job_queue 
                    WHERE job_type = 'neaps_ingestion' 
                    AND status IN ('pending', 'processing')
                    AND symbol IS NOT NULL
                    """
                )
                active = {}
                for symbol, status in cur.fetchall():
                    active[symbol] = status
                return active
        except Exception as exc:
            logger.error(f"[JobStore] get_active_neaps_symbols failed: {exc}")
            return {}

    def cleanup_old_jobs(self, retain_days: int = 30) -> int:
        """
        Delete terminal (completed / failed) jobs older than retain_days days.
        Call this on a nightly schedule or as part of the stale-job sweep.

        Returns:
            Number of rows deleted.
        """
        if not self._connect():
            return 0
        try:
            now = datetime.now(timezone.utc)
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    DELETE FROM job_queue
                    WHERE  status IN ('completed', 'failed')
                    AND    updated_at < (:now - NUMTODSINTERVAL(:days * 86400, 'SECOND'))
                    """,
                    {"now": now, "days": retain_days},
                )
                count = cur.rowcount
                conn.commit()
            if count:
                logger.info(f"[JobStore] Pruned {count} old job row(s) (retain={retain_days}d)")
            return count
        except Exception as exc:
            logger.error(f"[JobStore] cleanup_old_jobs failed: {exc}", exc_info=True)
            return 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_job_store: Optional[JobStore] = None
_store_lock = threading.Lock()          # guards singleton creation


def get_job_store() -> JobStore:
    """Get or create the shared JobStore singleton (thread-safe)."""
    global _job_store
    if _job_store is None:
        with _store_lock:
            if _job_store is None:      # double-checked locking
                _job_store = JobStore()
    return _job_store
