from __future__ import annotations

import logging
import os
import sys
import json
import time
from typing import Optional


class JsonFormatter(logging.Formatter):
    """
    Minimal JSON log formatter:
      { "t": 169, "lvl": "INFO", "name": "mod", "msg": "text", "extra": {...} }
    """

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "t": int(time.time() * 1000),
            "lvl": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        # Include extra dict if present
        if hasattr(record, "extra") and isinstance(record.extra, dict):  # type: ignore[attr-defined]
            payload["extra"] = record.extra  # type: ignore[attr-defined]
        # Include exception info if exists
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure root logger once with JSON formatting.
    Level precedence:
      - explicit `level` arg
      - env LOG_LEVEL (e.g., DEBUG/INFO/WARN/ERROR)
      - default INFO
    """
    root = logging.getLogger()
    if getattr(root, "_apnt_configured", False):  # idempotent
        return

    # Level
    lvl_name = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    try:
        lvl = getattr(logging, lvl_name)
    except AttributeError:
        lvl = logging.INFO

    # Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)
    root._apnt_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Get a module logger; ensures root is configured."""
    setup_logging()
    return logging.getLogger(name)
