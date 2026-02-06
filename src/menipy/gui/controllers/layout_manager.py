"""
Layout management for the main application.

Handles saving and restoring window geometry and state.
Extracted from MainController to adhere to Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


class LayoutManager:
    """Manages saving and restoring window layout."""

    def __init__(self, window: QMainWindow, settings):
        """Initialize.

        Parameters
        ----------
        window : type
        Description.
        settings : type
        Description.
        """
        self.window = window
        self.settings = settings

    def save_layout(self) -> None:
        """Save current window geometry and state to settings."""
        try:
            self.settings.main_window_geom_b64 = (
                self.window.saveGeometry().toBase64().data().decode("ascii")
            )
            self.settings.main_window_state_b64 = (
                self.window.saveState().toBase64().data().decode("ascii")
            )

            # Save splitter sizes if available
            splitter = getattr(self.window, "rootSplitter", None)
            if splitter:
                self.settings.splitter_sizes = splitter.sizes()

            self.settings.save()
            logger.info("Window state and settings saved.")
        except Exception as e:
            logger.error(f"Failed to save layout: {e}")

    def restore_layout(self) -> None:
        """Restore window geometry and state from settings."""
        try:
            geom_b64 = getattr(self.settings, "main_window_geom_b64", None)
            state_b64 = getattr(self.settings, "main_window_state_b64", None)

            if geom_b64:
                from PySide6.QtCore import QByteArray

                geom_bytes = QByteArray.fromBase64(geom_b64.encode("ascii"))
                self.window.restoreGeometry(geom_bytes)

            if state_b64:
                from PySide6.QtCore import QByteArray

                state_bytes = QByteArray.fromBase64(state_b64.encode("ascii"))
                self.window.restoreState(state_bytes)

            # Restore splitter sizes if available
            splitter = getattr(self.window, "rootSplitter", None)
            splitter_sizes = getattr(self.settings, "splitter_sizes", None)
            if splitter and splitter_sizes:
                splitter.setSizes(splitter_sizes)

            logger.info("Window layout restored from settings.")
        except Exception as e:
            logger.warning(f"Failed to restore layout: {e}")
