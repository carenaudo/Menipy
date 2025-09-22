"""Preview panel helper for Menipy GUI."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Callable

from PySide6.QtCore import QRectF, QLineF, Signal
from PySide6.QtWidgets import QWidget, QToolButton, QPushButton, QMainWindow
from PySide6.QtGui import QColor

from menipy.gui.views.image_view import DRAW_RECT, DRAW_LINE


class PreviewPanel:
    """Encapsulates the overlay/preview area interactions."""

    roi_selected = Signal(QRectF)
    line_drawn = Signal(QLineF)

    def __init__(
        self,
        window: QMainWindow,
        panel: QWidget,
        image_view_cls: Optional[type[Any]],
    ) -> None:
        self.window = window
        self.panel = panel
        self.image_view = getattr(window, "previewImageView", None)
        if not self.image_view and image_view_cls is not None:
            self.image_view = panel.findChild(image_view_cls, "previewView")

        self._on_roi_selected: Optional[Callable[[QRectF], None]] = None
        self._on_line_drawn: Optional[Callable[[QLineF], None]] = None

        if self.image_view:
            self._configure_image_view()
            self.image_view.roi_selected.connect(self.on_roi_selected)
            self.image_view.line_drawn.connect(self.on_line_drawn)

        self._wire_buttons()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_view(self) -> bool:
        return self.image_view is not None

    def load_path(self, path: str | Path) -> None:
        if not self.image_view:
            raise RuntimeError("Preview ImageView is unavailable")
        target = Path(path)
        if hasattr(self.image_view, "setImage"):
            self.image_view.setImage(str(target))
            return
        if hasattr(self.image_view, "load"):
            self.image_view.load(str(target))
            return
        raise RuntimeError("Preview ImageView has no loader method")

    def display(self, payload: Any) -> None:
        if not self.image_view:
            return
        if payload is None:
            return
        if hasattr(self.image_view, "set_image"):
            self.image_view.set_image(payload)
        elif hasattr(self.image_view, "setImage"):
            self.image_view.setImage(payload)
        elif hasattr(self.image_view, "load"):
            self.image_view.load(payload)

    def set_draw_mode(self, mode: Any, color: QColor = QColor(255, 0, 0)) -> None:
        if self.image_view and hasattr(self.image_view, "set_draw_mode"):
            self.image_view.set_draw_mode(mode, color)

    def clear_overlays(self) -> None:
        if self.image_view and hasattr(self.image_view, "clear_overlays"):
            self.image_view.clear_overlays()

    def set_roi_callback(self, handler: Optional[Callable[[QRectF], None]]) -> None:
        """Register callback invoked when ROI is drawn."""
        self._on_roi_selected = handler

    def set_line_callback(self, handler: Optional[Callable[[QLineF], None]]) -> None:
        """Register callback invoked when a contact line is drawn."""
        self._on_line_drawn = handler

    def on_roi_selected(self, rect: QRectF):
        if self._on_roi_selected:
            self._on_roi_selected(rect)

    def on_line_drawn(self, line: QLineF):
        if self._on_line_drawn:
            self._on_line_drawn(line)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _configure_image_view(self) -> None:
        try:
            self.image_view.set_auto_policy("preserve")
        except Exception:
            pass
        try:
            self.image_view.set_wheel_zoom_requires_ctrl(False)
        except Exception:
            pass

    def _wire_buttons(self) -> None:
        def _find_button(name: str) -> Optional[QToolButton | QPushButton]:
            button = self.panel.findChild(QToolButton, name)
            if button:
                return button
            return self.panel.findChild(QPushButton, name)

        if not self.image_view:
            return

        actions = {
            "roiBtn": lambda: self.set_draw_mode(DRAW_RECT, QColor(255, 255, 0)),  # Yellow
            "needleBtn": lambda: self.set_draw_mode(DRAW_RECT, QColor(0, 0, 255)),  # Blue
            "contactLineBtn": lambda: self.set_draw_mode(DRAW_LINE, QColor(173, 216, 230)),  # Light Blue
            "clearBtn": self.clear_overlays,
            "actualBtn": getattr(self.image_view, "actual_size", None),
            "fitBtn": getattr(self.image_view, "fit_to_window", None),
        }
        for name, handler in actions.items():
            if handler is None:
                continue
            button = _find_button(name)
            if button:
                try:
                    button.clicked.connect(handler)
                except Exception:
                    pass
