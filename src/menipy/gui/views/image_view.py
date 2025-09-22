from __future__ import annotations
from typing import Optional, Union
import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF, QSizeF, Signal, QLineF
from PySide6.QtGui import QImage, QPixmap, QTransform, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
)

# Drawing modes for overlays
DRAW_NONE = None
DRAW_POINT = "point"
DRAW_LINE = "line"
DRAW_RECT = "rect"


class ImageView(QGraphicsView):
    """QGraphicsView helper with pan/zoom and interactive overlays."""

    roi_selected = Signal(QRectF)
    line_drawn = Signal(QLineF)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- state (init first) ----
        self._scene = QGraphicsScene(self)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._mode: str = "auto"  # "fit" | "actual" | "free" | "auto"
        self._min_scale = 0.05
        self._max_scale = 20.0
        self._last_pm_size: Optional[QSizeF] = None
        self._auto_policy: str = "auto"  # "auto" | "preserve"
        self._wheel_zoom_requires_ctrl: bool = False

        self._draw_mode = DRAW_NONE
        self._overlays = []  # list[QGraphicsItem]
        self._tmp_item = None
        self._press_pos_scene: QPointF | None = None
        self._overlay_pen = QPen(QColor(255, 0, 0))
        self._overlay_pen.setWidth(2)
        # Ensure mouse tracking so we get move events even without press
        self.setMouseTracking(True)

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

    def set_draw_mode(self, mode: str | None, color: QColor = QColor(255, 0, 0)) -> None:
        """Update active drawing mode and overlay color."""

        self._draw_mode = mode
        self._overlay_pen.setColor(color)
        # When drawing, disable scroll-hand drag for comfort
        self.setDragMode(QGraphicsView.NoDrag if mode else QGraphicsView.ScrollHandDrag)

    def clear_overlays(self) -> None:
        for item in list(self._overlays):
            if not item:
                continue
            try:
                scene = item.scene()
            except RuntimeError:
                continue
            if scene:
                scene.removeItem(item)
        self._overlays.clear()
        self._tmp_item = None

    def set_image(self, img: Union[QPixmap, QImage, np.ndarray, None]) -> None:
        if img is None:
            self._scene.clear()
            self._pix_item = None
            self._last_pm_size = None
            return

        # Convert to QPixmap
        if isinstance(img, QPixmap):
            pm = img
            if pm.isNull():
                return
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
    def wheelEvent(self, event):
        # Require Ctrl only if configured
        if self._wheel_zoom_requires_ctrl and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            return super().wheelEvent(event)
        # Otherwise zoom w/o modifier
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep fit behavior on resize
        if getattr(self, "_mode", None) == "fit":
            self.fit_to_window()

    def _zoom_by(self, factor: float) -> None:
        if not self._pix_item:
            return
        scale_value = self.transform().m11()  # uniform scale
        new_scale = max(self._min_scale, min(self._max_scale, scale_value * factor))
        if abs(new_scale - scale_value) < 1e-6:
            return
        self.scale(new_scale / scale_value, new_scale / scale_value)
        self._mode = "free"

    def _fits_in_view(self, width: int, height: int) -> bool:
        viewport_width = max(1, self.viewport().width())
        viewport_height = max(1, self.viewport().height())
        return (width <= viewport_width) and (height <= viewport_height)

    # ---------------- Mouse handling for drawing ----------------
    def mousePressEvent(self, event):
        if self._draw_mode and event.button() == Qt.LeftButton and self.scene():
            scene_pos = self.mapToScene(event.pos())
            self._press_pos_scene = scene_pos
            if self._draw_mode == DRAW_POINT:
                radius = 4.0
                item = QGraphicsEllipseItem(
                    QRectF(scene_pos.x() - radius, scene_pos.y() - radius, 2 * radius, 2 * radius)
                )
                item.setPen(self._overlay_pen)
                item.setZValue(10_000)
                self.scene().addItem(item)
                self._overlays.append(item)
                self._press_pos_scene = None
            elif self._draw_mode == DRAW_LINE:
                item = QGraphicsLineItem(scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y())
                item.setPen(self._overlay_pen)
                item.setZValue(10_000)
                self.scene().addItem(item)
                self._tmp_item = item
            elif self._draw_mode == DRAW_RECT:
                item = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
                item.setPen(self._overlay_pen)
                item.setZValue(10_000)
                self.scene().addItem(item)
                self._tmp_item = item
            # Do not propagate to base if we're drawing
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._draw_mode and self._tmp_item and self._press_pos_scene and self.scene():
            current_pos = self.mapToScene(event.pos())
            if isinstance(self._tmp_item, QGraphicsLineItem):
                self._tmp_item.setLine(
                    self._press_pos_scene.x(),
                    self._press_pos_scene.y(),
                    current_pos.x(),
                    current_pos.y(),
                )
            elif isinstance(self._tmp_item, QGraphicsRectItem):
                rect = QRectF(self._press_pos_scene, current_pos).normalized()
                self._tmp_item.setRect(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._draw_mode and event.button() == Qt.LeftButton and self.scene():
            if self._tmp_item:
                if self._draw_mode == DRAW_RECT and isinstance(self._tmp_item, QGraphicsRectItem):
                    self.roi_selected.emit(self._tmp_item.rect())
                elif self._draw_mode == DRAW_LINE and isinstance(self._tmp_item, QGraphicsLineItem):
                    self.line_drawn.emit(self._tmp_item.line())

                # finalize
                self._overlays.append(self._tmp_item)
                self._tmp_item = None
                self._press_pos_scene = None
            return
        super().mouseReleaseEvent(event)
