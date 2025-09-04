from __future__ import annotations
from typing import Optional, Union
import numpy as np

from PySide6.QtCore import Qt, QRectF, QSizeF
from PySide6.QtGui import QImage, QPixmap, QTransform, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

class ImageView(QGraphicsView):
    """
    QGraphicsView with pan/zoom helpers:
      - set_image(QPixmap | QImage | np.ndarray[BGR])
      - zoom_in(), zoom_out(), actual_size(), fit_to_window()
      - Wheel zoom (Ctrl optional), hand-drag pan
      - Auto on load: 1:1 if it fits, else fit-to-window
      - 'preserve' policy keeps current pan/zoom on new frames of same size
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- state (init first) ----
        self._scene = QGraphicsScene(self)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._mode: str = "auto"                  # "fit" | "actual" | "free" | "auto"
        self._min_scale = 0.05
        self._max_scale = 20.0
        self._last_pm_size: Optional[QSizeF] = None
        self._auto_policy: str = "auto"           # "auto" | "preserve"
        self._wheel_zoom_requires_ctrl: bool = False

        # ---- view config ----
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints() | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # --------------------- public API ---------------------
    def set_auto_policy(self, policy: str = "auto") -> None:
        """'auto' (default): auto-fit/actual on new image; 'preserve': keep current pan/zoom if size unchanged."""
        if policy not in ("auto", "preserve"):
            raise ValueError("policy must be 'auto' or 'preserve'")
        self._auto_policy = policy

    def set_wheel_zoom_requires_ctrl(self, required: bool) -> None:
        self._wheel_zoom_requires_ctrl = bool(required)

    def set_image(self, img: Union[QPixmap, QImage, np.ndarray, None]) -> None:
        if img is None:
            self._scene.clear()
            self._pix_item = None
            self._last_pm_size = None
            return

        # Convert to QPixmap
        if isinstance(img, QPixmap):
            pm = img
        elif isinstance(img, QImage):
            pm = QPixmap.fromImage(img)
        else:
            # assume NumPy BGR
            rgb = img[..., ::-1].copy()
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qimg)

        # Preserve current transform/center if policy asks and size unchanged
        old_transform = QTransform(self.transform())
        old_center = self.mapToScene(self.viewport().rect().center())
        had_item = self._pix_item is not None
        same_size = had_item and (self._last_pm_size == QSizeF(pm.size()))

        self._scene.clear()
        self._pix_item = self._scene.addPixmap(pm)
        self._scene.setSceneRect(QRectF(pm.rect()))
        self._last_pm_size = QSizeF(pm.size())

        if self._auto_policy == "preserve" and same_size:
            # Keep current zoom/pan
            self.setTransform(old_transform)
            self.centerOn(old_center)
            self._mode = "free"
        else:
            # 1:1 if it fits in the viewport; otherwise fit-to-window (downscale only)
            self.resetTransform()
            if self._fits_in_view(pm.width(), pm.height()):
                self.actual_size()
            else:
                self.fit_to_window()

    def zoom_in(self, factor: float = 1.25) -> None:
        self._zoom_by(factor)

    def zoom_out(self, factor: float = 0.8) -> None:
        self._zoom_by(factor)

    def actual_size(self) -> None:
        if not self._pix_item:
            return
        self.setTransform(QTransform())  # identity
        self.centerOn(self._pix_item)
        self._mode = "actual"

    def fit_to_window(self) -> None:
        if not self._pix_item:
            return
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._mode = "fit"

    # --------------------- events & helpers ---------------------
    def wheelEvent(self, ev):
        # Require Ctrl only if configured
        if self._wheel_zoom_requires_ctrl and not (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return super().wheelEvent(ev)
        # Otherwise zoom w/o modifier
        if ev.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        ev.accept()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Keep fit behavior on resize
        if getattr(self, "_mode", None) == "fit":
            self.fit_to_window()

    def _zoom_by(self, factor: float):
        if not self._pix_item:
            return
        s = self.transform().m11()  # uniform scale
        new_s = max(self._min_scale, min(self._max_scale, s * factor))
        if abs(new_s - s) < 1e-6:
            return
        self.scale(new_s / s, new_s / s)
        self._mode = "free"

    def _fits_in_view(self, w: int, h: int) -> bool:
        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        return (w <= vw) and (h <= vh)
