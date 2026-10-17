"""JSON-lines logging to the console (for Docker/Kubernetes) and optionally to a file."""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.data_processing import PROJECT_ROOT

LOGGER_NAME = "heart_api"
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "api.log"


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to read, grep and ship to a log dashboard."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    """Create the API logger once. Set LOG_FILE="" to disable the log file."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:                       # already configured (e.g. reloads, tests)
        return logger
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    formatter = JsonFormatter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = os.environ.get("LOG_FILE", str(DEFAULT_LOG_FILE))
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:                       # read-only file system etc.: console only
            logger.warning("log_file_unavailable", extra={"fields": {"path": log_file}})
    return logger
