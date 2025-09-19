"""Qt logging bridge utilities for streaming Python logs into Qt widgets."""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QPlainTextEdit


class QtLogBridge(QObject):
    """Bridge object that exposes a Qt signal for log messages."""

    log = Signal(str)


class QtLogHandler(logging.Handler):
    """Logging handler that emits records via a Qt signal (thread-safe)."""

    def __init__(self, bridge: QtLogBridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        try:
            message = self.format(record)
            self.bridge.log.emit(message)
        except Exception:
            try:
                print(self.format(record))
            except Exception:
                pass


def install_qt_logging(
    log_view: QPlainTextEdit,
    formatter: Optional[logging.Formatter] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[QtLogBridge, QtLogHandler]:
    """Connect the global logger to a Qt text widget.

    Returns the created bridge and handler so callers can keep references.
    Subsequent calls detect existing handlers to avoid duplicates.
    """

    target_logger = logger or logging.getLogger()
    if formatter is None:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Reuse existing handler when present.
    for handler in target_logger.handlers:
        if isinstance(handler, QtLogHandler):
            bridge = handler.bridge
            bridge.log.connect(lambda message: log_view.appendPlainText(str(message)))
            return bridge, handler

    bridge = QtLogBridge()
    bridge.log.connect(lambda message: log_view.appendPlainText(str(message)))

    handler = QtLogHandler(bridge)
    handler.setFormatter(formatter)
    target_logger.addHandler(handler)
    return bridge, handler
