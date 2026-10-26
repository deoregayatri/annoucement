"""
Logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json

from config import settings


import logging.handlers

def setup_logging():
    """Setup application logging with distinct rotating files for production."""
    # Ensure log directory exists
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define log files
    app_log_file = settings.log_dir / "server.log"
    query_log_file = settings.log_dir / "queries.log"
    health_log_file = settings.log_dir / "health.log"
    
    # Create formatters
    if settings.log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # 1. Setup Root Logger (Server logs)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(settings.log_level))
    root_logger.handlers.clear()
    
    # Console handler (for docker/terminal output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)
    
    # Server Rotating File Handler (90 days retention)
    app_handler = logging.handlers.TimedRotatingFileHandler(
        app_log_file, when="midnight", interval=1, backupCount=90, encoding="utf-8"
    )
    app_handler.setLevel(logging.DEBUG)
    app_handler.setFormatter(formatter)
    root_logger.addHandler(app_handler)
    
    # 2. Setup Dedicated Query Logger
    query_logger = logging.getLogger("queries")
    query_logger.setLevel(logging.INFO)
    query_logger.propagate = False  # Don't send to root logger
    query_handler = logging.handlers.TimedRotatingFileHandler(
        query_log_file, when="midnight", interval=1, backupCount=90, encoding="utf-8"
    )
    query_handler.setFormatter(formatter)
    query_logger.addHandler(query_handler)

    # 3. Setup Dedicated Health Logger
    health_logger = logging.getLogger("health")
    health_logger.setLevel(logging.INFO)
    health_logger.propagate = False
    health_handler = logging.handlers.TimedRotatingFileHandler(
        health_log_file, when="midnight", interval=1, backupCount=90, encoding="utf-8"
    )
    health_handler.setFormatter(formatter)
    health_logger.addHandler(health_handler)
    
    # Silence verbose third-party loggers to prevent log spam
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured: server.log, queries.log, health.log (90-day retention)")


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)
