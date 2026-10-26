"""
Structured logging factory for the Announcement Intelligence system.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger pre-configured with the application log level.

    Args:
        name:  Module name, typically __name__.
        level: Override log level string (DEBUG/INFO/WARNING/ERROR/CRITICAL).
               Defaults to the value in settings.log_level.

    Returns:
        A standard Python Logger instance.
    """
    try:
        from config import settings
        effective_level = level or settings.log_level
    except Exception:
        effective_level = level or "INFO"

    logger = logging.getLogger(name)

    # Only configure if the logger has no handlers yet (avoid duplicating on re-import)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, effective_level.upper(), logging.INFO))
    logger.propagate = False
    return logger
