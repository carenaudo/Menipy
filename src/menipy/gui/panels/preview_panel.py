"""
Live preview panel for image display and interaction.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Any, Optional, Callable

from PySide6.QtCore import QRectF, QLineF, Signal
from PySide6.QtWidgets import QWidget, QToolButton, QPushButton
from PySide6.QtGui import QColor


class PreviewPanel:
    """Encapsulates the overlay/preview area interactions."""

    roi_selected = Signal(QRectF)
    line_drawn = Signal(QLineF)

    def __init__(
        self,
        panel: QWidget,
        image_view_cls: Optional[type[Any]],
    ) -> None:
        self.panel = panel
        self.image_view = getattr(panel, "previewView", None)
        if not self.image_view and image_view_cls is not None:
            self.image_view = panel.findChild(image_view_cls, "previewView")

        self._on_roi_selected: Optional[Callable[[QRectF], None]] = None
        self._on_line_drawn: Optional[Callable[[QLineF], None]] = None
        self._overlay_buttons: list[QToolButton | QPushButton] = []

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
        if hasattr(self.image_view, "set_image"):
            try:
                from PySide6.QtGui import QPixmap

                pixmap = QPixmap(str(target))
                if pixmap.isNull():
                    raise RuntimeError(f"Failed to load image: {target}")
                self.image_view.set_image(pixmap)
                logger.info("Preview loaded from %s", target)
                self._set_overlay_buttons_enabled(True)
                return
            except Exception as exc:
                raise RuntimeError(f"Failed to load image {target}: {exc}") from exc
        if hasattr(self.image_view, "setImage"):
            self.image_view.setImage(str(target))
            logger.info("Preview loaded from %s", target)
            self._set_overlay_buttons_enabled(True)
            return
        if hasattr(self.image_view, "load"):
            self.image_view.load(str(target))
            logger.info("Preview loaded from %s", target)
            self._set_overlay_buttons_enabled(True)
            return
        raise RuntimeError("Preview ImageView has no loader method")

    def display(self, payload: Any) -> None:
        if not self.image_view:
            return
        if payload is None:
            return
        handled = False
        if hasattr(self.image_view, "set_image"):
            self.image_view.set_image(payload, preserve_overlays=True)
            handled = True
        elif hasattr(self.image_view, "setImage"):
            self.image_view.setImage(payload)
            handled = True
        elif hasattr(self.image_view, "load"):
            self.image_view.load(payload)
            handled = True
        if handled:
            self._set_overlay_buttons_enabled(True)

    def set_draw_mode(self, mode: Any, color: QColor = QColor(255, 0, 0), *, tag: Optional[str] = None) -> None:
        if self.image_view and hasattr(self.image_view, "set_draw_mode"):
            try:
                self.image_view.set_draw_mode(mode, color, tag=tag)
            except TypeError:
                self.image_view.set_draw_mode(mode, color)
                if hasattr(self.image_view, "set_draw_tag"):
                    self.image_view.set_draw_tag(tag)

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


    def has_roi(self) -> bool:
        return self.roi_rect() is not None

    def has_needle(self) -> bool:
        return self.needle_rect() is not None

    def has_contact_line(self) -> bool:
        return self.contact_line_segment() is not None

    def roi_rect(self) -> tuple[int, int, int, int] | None:
        rect = self._overlay_rect('roi')
        if rect is None:
            return None
        rect = rect.normalized()
        width = rect.width()
        height = rect.height()
        if width <= 0 or height <= 0:
            return None
        return (
            int(round(rect.left())),
            int(round(rect.top())),
            int(round(rect.width())),
            int(round(rect.height())),
        )

    def needle_rect(self) -> tuple[int, int, int, int] | None:
        rect = self._overlay_rect('needle')
        if rect is None:
            return None
        rect = rect.normalized()
        width = rect.width()
        height = rect.height()
        if width <= 0 or height <= 0:
            return None
        return (
            int(round(rect.left())),
            int(round(rect.top())),
            int(round(rect.width())),
            int(round(rect.height())),
        )

    def contact_line_segment(self) -> tuple[tuple[int, int], tuple[int, int]] | None:
        line = self._overlay_line('contact_line')
        if line is None:
            return None
        p1 = line.p1()
        p2 = line.p2()
        if p1 == p2:
            return None
        return (
            (int(round(p1.x())), int(round(p1.y()))),
            (int(round(p2.x())), int(round(p2.y()))),
        )

    def _overlay_rect(self, tag: str) -> QRectF | None:
        if not self.image_view or not hasattr(self.image_view, 'overlay_rect_scene'):
            return None
        return self.image_view.overlay_rect_scene(tag)

    def _overlay_line(self, tag: str) -> QLineF | None:
        if not self.image_view or not hasattr(self.image_view, 'overlay_line_scene'):
            return None
        return self.image_view.overlay_line_scene(tag)

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

        try:
            from menipy.gui.views.image_view import DRAW_RECT, DRAW_LINE
        except ImportError:
            return

        actions = {
            "roiBtn": lambda: self.set_draw_mode(DRAW_RECT, QColor(255, 255, 0), tag="roi"),  # Yellow
            "needleBtn": lambda: self.set_draw_mode(DRAW_RECT, QColor(0, 0, 255), tag="needle"),  # Blue
            "contactLineBtn": lambda: self.set_draw_mode(DRAW_LINE, QColor(173, 216, 230), tag="contact_line"),  # Light Blue
            "clearBtn": self.clear_overlays,
            "actualBtn": getattr(self.image_view, "actual_size", None),
            "fitBtn": getattr(self.image_view, "fit_to_window", None),
        }
        overlay_names = {"roiBtn", "needleBtn", "contactLineBtn"}
        for name, handler in actions.items():
            if handler is None:
                continue
            button = _find_button(name)
            if button:
                if name in overlay_names:
                    self._overlay_buttons.append(button)
                try:
                    button.clicked.connect(handler)
                except Exception:
                    pass
        self._set_overlay_buttons_enabled(False)

    def _set_overlay_buttons_enabled(self, enabled: bool) -> None:
        if not self._overlay_buttons:
            return
        for button in self._overlay_buttons:
            try:
                button.setEnabled(enabled)
            except Exception:
                pass
