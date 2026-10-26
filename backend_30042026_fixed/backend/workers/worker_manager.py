"""
WorkerManager - Stage 4 (Optimized).

Background daemon that polls Oracle job_queue and dispatches handlers.

Architecture:
  - One daemon polling thread (adaptive interval: 0.2s busy / 5s idle)
  - ThreadPoolExecutor  for I/O-bound jobs  (json_fetch, excel_ingestion)
  - ProcessPoolExecutor for CPU-heavy jobs  (document_ingestion)
    · Uses 'spawn' multiprocessing context (avoids model-reload on fork)
    · initializer=_cpu_worker_init loads bge-large ONCE per subprocess
    · Worker count computed from available RAM + CPU cores at startup
    · BrokenProcessPool auto-restart on subprocess crash

Backpressure:
  - BackpressureController checks CPU% + RAM% + pending queue size
  - Halves CPU slots at CPU>78% or MEM>78%; pauses at CPU>92% or MEM>88%

Observability:
  - Per-job queue_wait_ms (time from created_at → claimed)
  - Batch report (total time, succeeded, failed)

Lifecycle:
  - start()  called from FastAPI lifespan on startup
  - stop()   called from FastAPI lifespan on shutdown
"""

import concurrent.futures
import logging
import os
import socket
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Job types routed to ProcessPoolExecutor (CPU-heavy)
_CPU_JOB_TYPES = {"document_ingestion", "neaps_ingestion"}

# Seconds between stale-job requeue sweeps
REQUEUE_INTERVAL_S = 1800


# ---------------------------------------------------------------------------
# Memory-aware worker count
# ---------------------------------------------------------------------------


def _get_linux_free_ram_gb() -> Optional[float]:
    """Read available RAM directly from /proc/meminfo on Linux without requiring psutil."""
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # Value in kB
                        kb = float(line.split()[1])
                        return kb / (1024 * 1024)
    except Exception:
        pass
    return None


def _compute_worker_count(
    ram_per_worker_gb: float = 1.6,
    system_reserve_gb: float = 2.0,
    cpu_headroom: float = 0.75,
    hard_cap: Optional[int] = None,
    min_workers: int = 1,
) -> int:
    """
    Compute safe ProcessPool worker count from available RAM + CPU cores.
    Always strictly respects settings.max_workers_cpu from .env as the upper limit.
    Does NOT require psutil.
    """
    from config import settings
    env_workers = max(1, int(getattr(settings, "max_workers_cpu", 2)))
    cap = env_workers if hard_cap is None else min(hard_cap, env_workers)

    free_gb = None
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1024**3
    except ImportError:
        # Fallback to native Linux /proc/meminfo (no package required)
        free_gb = _get_linux_free_ram_gb()
    except Exception:
        free_gb = None

    if free_gb is not None:
        try:
            cpu_cores = os.cpu_count() or 4
            by_ram = int((free_gb - system_reserve_gb) / ram_per_worker_gb)
            by_cpu = int(cpu_cores * cpu_headroom)
            result = max(min_workers, min(by_ram, by_cpu, cap))
            logger.info(
                f"[WorkerCalc] free_RAM={free_gb:.1f}GB cpu={cpu_cores} "
                f"by_ram={by_ram} by_cpu={by_cpu} max_workers_cpu={env_workers} → workers={result}"
            )
            return result
        except Exception as exc:
            logger.warning(f"[WorkerCalc] Error in calculation: {exc} — using {env_workers}")
            return env_workers
    else:
        logger.info(
            f"[WorkerCalc] psutil not installed — strictly using MAX_WORKERS_CPU={env_workers} from .env"
        )
        return env_workers


# ---------------------------------------------------------------------------
# Backpressure controller
# ---------------------------------------------------------------------------


class BackpressureController:
    """
    Called on every poll cycle.  Reports how many CPU slots are allowed
    and whether to accept new jobs at all, based on system load.
    """

    CPU_PAUSE = 92  # %  — stop accepting new CPU jobs
    CPU_REDUCE = 78  # %  — reduce CPU slots by 1
    MEM_PAUSE = 88  # %  — stop accepting any new jobs
    MEM_REDUCE = 78  # %  — reduce CPU slots by 1
    QUEUE_FAST = 50  # pending jobs above which → fast poll (0.2s)

    def evaluate(self, configured_cpu_max: int, pending_count: int) -> dict:
        """
        Returns:
          cpu_slots  : int   — CPU job slots to allow this cycle
          accept     : bool  — False = pause job intake
          poll_next  : float — seconds until next poll cycle
        """
        try:
            import psutil

            cpu = psutil.cpu_percent(interval=0.15)
            mem = psutil.virtual_memory().percent
        except Exception:
            return {"cpu_slots": configured_cpu_max, "accept": True, "poll_next": 5.0}

        # Hard stop
        if mem >= self.MEM_PAUSE:
            logger.warning(
                f"[Backpressure] MEM={mem:.0f}% ≥ {self.MEM_PAUSE}% → PAUSE ALL"
            )
            return {"cpu_slots": 0, "accept": False, "poll_next": 15.0}
        if cpu >= self.CPU_PAUSE:
            logger.warning(
                f"[Backpressure] CPU={cpu:.0f}% ≥ {self.CPU_PAUSE}% → PAUSE CPU"
            )
            return {"cpu_slots": 0, "accept": False, "poll_next": 10.0}

        # Soft reduce
        slots = configured_cpu_max
        if cpu >= self.CPU_REDUCE or mem >= self.MEM_REDUCE:
            slots = max(1, configured_cpu_max - 1)
            logger.debug(
                f"[Backpressure] CPU={cpu:.0f}% MEM={mem:.0f}% → reduce to {slots} slots"
            )

        # Adaptive poll interval
        if pending_count >= self.QUEUE_FAST:
            poll_next = 0.2  # aggressive drain when queue is large
        elif pending_count > 0:
            poll_next = 0.5  # normal active
        else:
            poll_next = 5.0  # idle

        return {"cpu_slots": slots, "accept": True, "poll_next": poll_next}


# ---------------------------------------------------------------------------
# WorkerManager
# ---------------------------------------------------------------------------


class WorkerManager:
    """
    Polls Oracle job_queue and dispatches jobs to thread/process pools.

    Thread safety:
      The poll loop runs in a dedicated daemon thread.  Individual job futures
      are tracked per executor; completions are processed on the next poll cycle.
    """

    def __init__(
        self,
        poll_interval_s: int = 5,
        max_workers_io: int = 4,
        max_workers_cpu: int = 2,  # overridden by _compute_worker_count() at start()
        lock_timeout_s: int = 3600,
        heartbeat_interval_s: int = 300,
        heartbeat_timeout_s: int = 900,
        jobs_per_poll: int = 10,
    ) -> None:
        self.poll_interval_s = poll_interval_s
        self.max_workers_io = max_workers_io
        self.max_workers_cpu = max_workers_cpu
        self.lock_timeout_s = lock_timeout_s
        self.heartbeat_interval_s = heartbeat_interval_s
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.jobs_per_poll = jobs_per_poll

        # Batch report tracking
        self._batch_start_time: Optional[float] = None
        self._batch_completed = 0
        self._batch_failed = 0
        self._was_processing = False

        self.worker_id = f"{socket.gethostname()}-{threading.get_ident()}"

        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._io_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._cpu_executor: Optional[concurrent.futures.ProcessPoolExecutor] = None

        self._futures: Dict[concurrent.futures.Future, Any] = {}
        self._heartbeats: Dict[concurrent.futures.Future, threading.Event] = {}
        self._lock = threading.Lock()

        self._handler_map: Optional[Dict[str, Callable]] = None
        self._backpressure = BackpressureController()

        logger.info(
            f"[WorkerManager] Configured: poll={poll_interval_s}s "
            f"io_workers={max_workers_io} cpu_workers={max_workers_cpu}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background poll loop and executor pools."""
        if self._poll_thread and self._poll_thread.is_alive():
            logger.warning(
                "[WorkerManager] Already running — ignoring duplicate start()."
            )
            return

        self._stop_event.clear()

        # IO pool (fixed — json_fetch is network-bound)
        self._io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers_io,
            thread_name_prefix="jq-io",
        )

        # CPU pool — compute safe worker count dynamically
        self.max_workers_cpu = _compute_worker_count()
        self._create_cpu_executor()

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="jq-poll",
            daemon=True,
        )
        self._poll_thread.start()
        logger.info(f"[WorkerManager] Started — worker_id={self.worker_id}")

    def _create_cpu_executor(self) -> None:
        """
        Create the ProcessPoolExecutor with the spawn start method and
        the subprocess model initializer.

        'spawn' is used explicitly on ALL platforms to ensure:
          1. No fork-inherited state (avoids file descriptor / lock leaks)
          2. Embedding model is loaded fresh in each subprocess via initializer=
          3. Consistent behaviour between Linux (where fork is the default) and Windows
        """
        if os.name == "nt":
            # Windows: ProcessPoolExecutor with spawn is the native default.
            logger.info(
                f"[WorkerManager] CPU pool: ProcessPoolExecutor "
                f"workers={self.max_workers_cpu} context=spawn (Windows)"
            )
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")
        else:
            logger.info(
                f"[WorkerManager] CPU pool: ProcessPoolExecutor "
                f"workers={self.max_workers_cpu} context=spawn (Linux)"
            )
            import multiprocessing

            ctx = multiprocessing.get_context("spawn")

        # Import the subprocess initializer here so it is picklable
        from ingestion.neaps_ingestion_service import _subprocess_init

        self._cpu_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers_cpu,
            mp_context=ctx,
            initializer=_subprocess_init,
            max_tasks_per_child=50,  # Recycle workers every 50 PDFs to flush leaked RAM
        )
        logger.info(
            f"[WorkerManager] ProcessPoolExecutor created: "
            f"workers={self.max_workers_cpu} spawn+initializer max_tasks_per_child=50"
        )

    def _restart_cpu_executor(self) -> None:
        """
        Restart the CPU executor after a BrokenProcessPool event.
        Called from _dispatch() when BrokenProcessPool is caught.
        """
        logger.critical(
            "[WorkerManager] BrokenProcessPool detected — restarting CPU executor..."
        )
        try:
            self._cpu_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.max_workers_cpu = _compute_worker_count()
        self._create_cpu_executor()
        logger.info(
            f"[WorkerManager] CPU executor restarted ({self.max_workers_cpu} workers)."
        )

    def stop(self, wait_s: float = 30.0) -> None:
        """Signal shutdown and wait for in-flight jobs to complete."""
        logger.info("[WorkerManager] Shutdown requested ...")
        self._stop_event.set()

        if self._poll_thread:
            self._poll_thread.join(timeout=wait_s)

        if self._io_executor:
            self._io_executor.shutdown(wait=True, cancel_futures=False)
        if self._cpu_executor:
            self._cpu_executor.shutdown(wait=True, cancel_futures=False)

        logger.info("[WorkerManager] Stopped cleanly.")

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Main daemon loop: claim → dispatch → harvest completions."""
        from ingestion.job_store import get_job_store

        store = get_job_store()

        last_requeue_check = time.monotonic()
        last_worker_recompute = time.monotonic()
        RECOMPUTE_INTERVAL_S = 600  # re-evaluate worker count every 10 min

        while not self._stop_event.is_set():
            try:
                # 1. Harvest completed futures
                self._harvest_futures(store)

                # 2. Stale-job sweep (every REQUEUE_INTERVAL_S)
                if time.monotonic() - last_requeue_check > REQUEUE_INTERVAL_S:
                    store.requeue_stale_jobs(self.heartbeat_timeout_s)
                    store.cleanup_old_jobs(retain_days=30)
                    last_requeue_check = time.monotonic()

                # 3. Re-evaluate worker count (every 10 min, adapts to RAM changes)
                if time.monotonic() - last_worker_recompute > RECOMPUTE_INTERVAL_S:
                    new_count = _compute_worker_count()
                    if new_count != self.max_workers_cpu:
                        logger.info(
                            f"[WorkerManager] Worker count updated: "
                            f"{self.max_workers_cpu} → {new_count}"
                        )
                        self.max_workers_cpu = new_count
                    last_worker_recompute = time.monotonic()

                # 4. Backpressure check
                pending_count = self._get_pending_count(store)
                bp = self._backpressure.evaluate(self.max_workers_cpu, pending_count)

                if not bp["accept"]:
                    self._stop_event.wait(timeout=bp["poll_next"])
                    continue

                # 5. Claim and dispatch new jobs
                slots = self._available_slots(cpu_override=bp["cpu_slots"])
                jobs_claimed = []
                if slots > 0:
                    jobs_claimed = store.claim_jobs(n=slots, worker_id=self.worker_id)
                    if jobs_claimed:
                        if not self._was_processing:
                            self._was_processing = True
                            self._batch_start_time = time.time()
                            self._batch_completed = 0
                            self._batch_failed = 0
                            logger.info(
                                "[WorkerManager] [START] New job batch started."
                            )

                        for job in jobs_claimed:
                            self._dispatch(job)

                # 6. Detect batch completion
                with self._lock:
                    in_flight = len(self._futures)
                if self._was_processing and in_flight == 0:
                    self._print_ingestion_report()
                    self._was_processing = False

            except Exception as exc:
                logger.error(f"[WorkerManager] Poll loop error: {exc}", exc_info=True)

            # Adaptive poll interval
            had_jobs = bool(jobs_claimed) if "jobs_claimed" in dir() else False
            self._stop_event.wait(
                timeout=bp.get("poll_next", self.poll_interval_s)
                if "bp" in dir()
                else self.poll_interval_s
            )

        logger.debug("[WorkerManager] Poll loop exited.")

    @staticmethod
    def _get_pending_count(store: Any) -> int:
        """Safe wrapper around get_pending_count()."""
        try:
            return store.get_pending_count()
        except Exception:
            return 0

    def _available_slots(self, cpu_override: Optional[int] = None) -> int:
        """Return how many additional jobs we can accept right now."""
        cpu_max = cpu_override if cpu_override is not None else self.max_workers_cpu
        with self._lock:
            in_flight_io = sum(
                1 for j in self._futures.values() if j.job_type not in _CPU_JOB_TYPES
            )
            in_flight_cpu = sum(
                1 for j in self._futures.values() if j.job_type in _CPU_JOB_TYPES
            )
        io_slots = max(0, self.max_workers_io - in_flight_io)
        cpu_slots = max(0, cpu_max - in_flight_cpu)
        return io_slots + cpu_slots

    def _start_heartbeat(self, job_id: str) -> threading.Event:
        stop_event = threading.Event()
        def _beat():
            while not stop_event.wait(timeout=self.heartbeat_interval_s):
                from ingestion.job_store import get_job_store
                alive = get_job_store().heartbeat(job_id, self.worker_id)
                if not alive:
                    logger.warning(f"[Heartbeat] job_id={job_id[:8]} was stolen or lost!")
        threading.Thread(target=_beat, daemon=True, name=f"hb-{job_id[:8]}").start()
        return stop_event

    def _dispatch(self, job: Any) -> None:
        """Submit a job to the appropriate executor."""
        with self._lock:
            in_flight_ids = {j.job_id for j in self._futures.values()}
        if job.job_id in in_flight_ids:
            logger.warning(
                f"[WorkerManager] DUPLICATE DISPATCH BLOCKED: "
                f"job_id={job.job_id[:8]} is already in-flight — skipping."
            )
            return

        handler = self._resolve_handler(job.job_type)
        if handler is None:
            logger.error(
                f"[WorkerManager] Unknown job_type='{job.job_type}' — skipping."
            )
            from ingestion.job_store import get_job_store

            get_job_store().mark_failed(
                job.job_id,
                f"No handler for job_type='{job.job_type}'",
                job.retry_count,
                job.max_retries,
            )
            return

        is_cpu = job.job_type in _CPU_JOB_TYPES
        executor = self._cpu_executor if is_cpu else self._io_executor

        try:
            future = executor.submit(handler, job.payload)
            with self._lock:
                self._futures[future] = job
                self._heartbeats[future] = self._start_heartbeat(job.job_id)
            logger.info(
                f"[WorkerManager] Dispatched job_id={job.job_id[:8]} "
                f"type={job.job_type} retry={job.retry_count}"
            )
        except concurrent.futures.process.BrokenProcessPool:
            logger.critical(
                f"[WorkerManager] BrokenProcessPool on dispatch of {job.job_id[:8]}"
            )
            self._restart_cpu_executor()
            try:
                future = self._cpu_executor.submit(handler, job.payload)
                with self._lock:
                    self._futures[future] = job
                    self._heartbeats[future] = self._start_heartbeat(job.job_id)
                logger.info(
                    f"[WorkerManager] Re-dispatched {job.job_id[:8]} after pool restart."
                )
            except Exception as retry_exc:
                logger.error(f"[WorkerManager] Re-dispatch failed: {retry_exc}")
                from ingestion.job_store import get_job_store

                get_job_store().mark_failed(
                    job.job_id, str(retry_exc), job.retry_count, job.max_retries
                )
        except Exception as exc:
            logger.error(f"[WorkerManager] Dispatch failed for {job.job_id[:8]}: {exc}")
            from ingestion.job_store import get_job_store

            get_job_store().mark_failed(
                job.job_id, str(exc), job.retry_count, job.max_retries
            )

    def _harvest_futures(self, store: Any) -> None:
        """Check completed futures and update job statuses."""
        with self._lock:
            done = {f: j for f, j in self._futures.items() if f.done()}
            for f in done:
                del self._futures[f]
                stop_event = self._heartbeats.pop(f, None)
                if stop_event:
                    stop_event.set()

        for future, job in done.items():
            exc = future.exception()
            if exc is None:
                store.mark_completed(job.job_id)
                self._batch_completed += 1
                logger.info(
                    f"[WorkerManager] Completed job_id={job.job_id[:8]} type={job.job_type}"
                )
            else:
                error_str = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                logger.error(
                    f"[WorkerManager] Job failed job_id={job.job_id[:8]} "
                    f"type={job.job_type}: {exc}"
                )
                self._batch_failed += 1
                store.mark_failed(
                    job.job_id,
                    str(exc)[:2000],
                    job.retry_count,
                    job.max_retries,
                )

    def _print_ingestion_report(self) -> None:
        """Triggered when the queue naturally drops to exactly 0."""
        elapsed = time.time() - (self._batch_start_time or time.time())
        mins, s = divmod(int(elapsed), 60)
        hrs, m = divmod(mins, 60)
        time_str = f"{hrs}h {m}m {s}s" if hrs > 0 else f"{m}m {s}s"
        total = self._batch_completed + self._batch_failed

        logger.info("\n" + "=" * 60)
        logger.info("[REPORT] BATCH INGESTION REPORT (COMPLETED)")
        logger.info("=" * 60)
        logger.info(f"Total Time Taken : {time_str}")
        logger.info(f"Total Processed  : {total} items")
        logger.info(f"[OK]     Succeeded: {self._batch_completed}")
        logger.info(f"[FAILED] Failed   : {self._batch_failed}")
        logger.info("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Handler resolution
    # ------------------------------------------------------------------

    def _get_handler_map(self) -> Dict[str, Callable]:
        if self._handler_map is not None:
            return self._handler_map
        try:
            from workers.handlers import (
                handle_excel_ingestion,
                handle_json_fetch,
                handle_document_ingestion,
                handle_vlm_image_processing,
                handle_neaps_ingestion,
            )

            self._handler_map = {
                "excel_ingestion": handle_excel_ingestion,
                "json_fetch": handle_json_fetch,
                "document_ingestion": handle_document_ingestion,
                "vlm_image_processing": handle_vlm_image_processing,
                "neaps_ingestion": handle_neaps_ingestion,
            }
        except Exception as exc:
            logger.error(f"[WorkerManager] Handler import failed: {exc}")
            self._handler_map = {}
        return self._handler_map

    def _resolve_handler(self, job_type: str) -> Optional[Callable]:
        return self._get_handler_map().get(job_type)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_worker_manager: Optional[WorkerManager] = None


def get_worker_manager() -> WorkerManager:
    """Return the shared WorkerManager singleton, configured from settings."""
    global _worker_manager
    if _worker_manager is None:
        try:
            from config import settings

            _worker_manager = WorkerManager(
                poll_interval_s=getattr(settings, "worker_poll_interval_s", 5),
                max_workers_io=getattr(settings, "max_workers_io", 4),
                max_workers_cpu=getattr(settings, "max_workers_cpu", 2),
                lock_timeout_s=getattr(settings, "worker_lock_timeout_s", 259200),
                heartbeat_interval_s=getattr(settings, "worker_heartbeat_interval_s", 300),
                heartbeat_timeout_s=getattr(settings, "worker_heartbeat_timeout_s", 900),
            )
        except Exception:
            _worker_manager = WorkerManager()
    return _worker_manager
 