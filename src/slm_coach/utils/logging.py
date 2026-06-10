"""Structured logging for the pipeline.

A single :func:`configure_logging` call sets up the root logger with either a rich,
human-friendly console handler or a JSON formatter (for machine ingestion). Library
code should obtain loggers via :func:`get_logger` and never call :func:`print`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.logging import RichHandler

_CONFIGURED = False
_LOGGER_NAME = "slm_coach"


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects.

    Any non-standard attributes attached to a record via ``logger.info(msg, extra=...)``
    are merged into the emitted object, which makes structured fields easy to grep.
    """

    _RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Serialize ``record`` to a JSON string."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | int = "INFO", *, json_logs: bool = False) -> None:
    """Configure the package logger exactly once.

    Args:
        level: Logging level name (e.g. ``"INFO"``) or numeric level.
        json_logs: If ``True``, emit JSON lines; otherwise use a rich console handler.
    """
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    if _CONFIGURED:
        logger.setLevel(level)
        return

    handler: logging.Handler
    if json_logs:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))

    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``slm_coach`` namespace.

    Args:
        name: Optional dotted suffix (typically ``__name__``). When ``None`` the root
            package logger is returned.

    Returns:
        A configured :class:`logging.Logger`.
    """
    if not _CONFIGURED:
        configure_logging()
    if not name or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    suffix = name.split(".")[-1] if name.startswith(_LOGGER_NAME) else name
    return logging.getLogger(f"{_LOGGER_NAME}.{suffix}")
