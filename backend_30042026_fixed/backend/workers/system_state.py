"""
Shared system state for signalling busy/idle to lightweight endpoints.
Thread-safe singleton used by ingestion and health check.
"""

import threading
import time
from typing import Optional, Tuple


class SystemState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy_count: int = 0
        self._reason: str = ""
        self._since: Optional[float] = None

    def set_busy(self, reason: str = "") -> None:
        with self._lock:
            self._busy_count += 1
            self._reason = reason or "busy"
            if self._since is None:
                self._since = time.time()

    def clear_busy(self) -> None:
        with self._lock:
            if self._busy_count > 0:
                self._busy_count -= 1
            if self._busy_count == 0:
                self._reason = ""
                self._since = None

    def is_busy(self) -> Tuple[bool, str, Optional[float]]:
        with self._lock:
            return self._busy_count > 0, self._reason, self._since


_system_state: Optional[SystemState] = None


def get_system_state() -> SystemState:
    global _system_state
    if _system_state is None:
        _system_state = SystemState()
    return _system_state
