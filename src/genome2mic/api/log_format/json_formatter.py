"""One-JSON-object-per-line log formatter."""

import json
import logging
import sys
from datetime import UTC, datetime

from genome2mic.api.request_context import request_id_var


class JsonFormatter(logging.Formatter):
    """Formats each log record as one JSON line that carries the request id and any `extra` fields."""

    _STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName", "color_message"}

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
        }
        for key, value in vars(record).items():
            if key not in self._STANDARD_ATTRS and key not in entry:
                entry[key] = value
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)

    @staticmethod
    def install(level: str) -> None:
        """Send all logs, uvicorn's included, to stdout as JSON."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root_logger = logging.getLogger()
        root_logger.handlers = [handler]
        root_logger.setLevel(level)
        for name in ("uvicorn", "uvicorn.error"):
            uvicorn_logger = logging.getLogger(name)
            uvicorn_logger.handlers = []
            uvicorn_logger.propagate = True
        # RequestIdMiddleware already logs every request with its id.
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.handlers = []
        access_logger.propagate = False
