"""Structured (key=value) logging setup. No secrets or PII are ever logged."""
from __future__ import annotations

import logging


class _KeyValueFormatter(logging.Formatter):
    _RESERVED = set(logging.makeLogRecord({}).__dict__.keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        base = f"ts={self.formatTime(record)} level={record.levelname} " \
               f"logger={record.name} event={record.getMessage()}"
        extras = {
            k: v for k, v in record.__dict__.items() if k not in self._RESERVED
        }
        if extras:
            kv = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base} {kv}"
        return base


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_KeyValueFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
