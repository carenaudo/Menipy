from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
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



def _enum_to_int(value: object) -> int:
    """Convert Qt enum/flag objects to plain ints for signal payloads."""
    if hasattr(value, 'value'):
        try:
            return int(value.value)
        except (TypeError, ValueError):
            pass
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0

class ImageView(QGraphicsView):
    """QGraphicsView helper with pan/zoom and interactive overlays."""

    roi_selected = Signal(QRectF)
    line_drawn = Signal(QLineF)
    point_clicked = Signal(QPointF, int, int)
    double_clicked = Signal(QPointF, int, int)

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
        self._overlay_items_by_tag: dict[str, object] = {}
        self._draw_tag: str | None = None
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

    def set_draw_mode(self, mode: str | None, color: QColor = QColor(255, 0, 0), tag: str | None = None) -> None:
        """Update active drawing mode and overlay color."""

        self._draw_mode = mode
        self._draw_tag = tag if mode else None
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
        self._overlay_items_by_tag.clear()
        self._tmp_item = None

    def remove_overlay(self, tag: str) -> None:
        item = self._overlay_items_by_tag.get(tag)
        if not item:
            return
        self._remove_overlay_item(item)

    def add_marker_point(
        self,
        point: QPointF,
        *,
        color: QColor = QColor(0, 255, 0),
        radius: float = 4.0,
        alpha: float | None = None,
        stroke_width: float = 2.0,
        shape: str = "circle",  # circle | square | cross
        dash_pattern: tuple[float, float] | None = None,
        drop_shadow: bool = False,
        tag: str | None = None,
    ) -> None:
        # Create shape-specific QGraphicsItem
        if shape == "square":
            item = QGraphicsRectItem(QRectF(point.x() - radius, point.y() - radius, 2 * radius, 2 * radius))
        elif shape == "cross":
            # use a small group: two lines
            l1 = QGraphicsLineItem(point.x() - radius, point.y() - radius, point.x() + radius, point.y() + radius)
            l2 = QGraphicsLineItem(point.x() - radius, point.y() + radius, point.x() + radius, point.y() - radius)
            pen = QPen(color)
            pen.setWidthF(float(stroke_width))
            if dash_pattern is not None:
                try:
                    pen.setDashPattern([float(dash_pattern[0]), float(dash_pattern[1])])
                except Exception:
                    pass
            if alpha is not None:
                try:
                    c = QColor(color)
                    c.setAlphaF(max(0.0, min(1.0, float(alpha))))
                    pen.setColor(c)
                except Exception:
                    pass
            l1.setPen(pen)
            l2.setPen(pen)
            l1.setZValue(12_000)
            l2.setZValue(12_000)
            self.scene().addItem(l1)
            self.scene().addItem(l2)
            self._overlays.extend([l1, l2])
            self._register_overlay_item(l1, tag)
            return
        else:
            item = QGraphicsEllipseItem(
                QRectF(point.x() - radius, point.y() - radius, 2 * radius, 2 * radius)
            )

        pen = QPen(color)
        pen.setWidthF(float(stroke_width))
        if alpha is not None:
            try:
                c = QColor(color)
                c.setAlphaF(max(0.0, min(1.0, float(alpha))))
                pen.setColor(c)
                item.setBrush(c)
            except Exception:
                item.setBrush(color)
        else:
            item.setBrush(color)
        item.setPen(pen)
        item.setZValue(12_000)
        self.scene().addItem(item)
        if drop_shadow:
            try:
                from PySide6.QtWidgets import QGraphicsDropShadowEffect
                eff = QGraphicsDropShadowEffect()
                eff.setBlurRadius(8)
                eff.setOffset(2, 2)
                # attach effect to the underlying widget where appropriate
                # QGraphicsItems don't directly accept QGraphicsEffects, so this is best-effort
                # If framework doesn't accept it, we ignore silently.
                item.setGraphicsEffect(eff)
            except Exception:
                pass
        self._overlays.append(item)
        self._register_overlay_item(item, tag)

    def add_marker_line(
        self,
        p1: QPointF,
        p2: QPointF,
        *,
        color: QColor = QColor(255, 140, 0),
        tag: str | None = None,
    ) -> None:
        item = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
        pen = QPen(color)
        pen.setWidth(2)
        item.setPen(pen)
        item.setZValue(11_000)
        self.scene().addItem(item)
        self._overlays.append(item)
        self._register_overlay_item(item, tag)

    def add_marker_contour(
        self,
        pts: np.ndarray,
        *,
        color: QColor = QColor(255, 0, 0),
        width: float = 2.0,
        dash_pattern: tuple[float, float] | None = None,
        alpha: float | None = None,
        tag: str | None = None,
    ) -> None:
        """Add a contour path overlay from an Nx2 array of (x,y) points.

        The contour is added as a single QGraphicsPathItem and registered under `tag`.
        """
        try:
            if pts is None:
                return
            pts = np.asarray(pts, dtype=float)
            if pts.ndim != 2 or pts.shape[1] < 2:
                return
            from PySide6.QtGui import QPainterPath
            path = QPainterPath()
            path.moveTo(QPointF(float(pts[0, 0]), float(pts[0, 1])))
            for p in pts[1:]:
                path.lineTo(QPointF(float(p[0]), float(p[1])))
            # close path
            path.lineTo(QPointF(float(pts[0, 0]), float(pts[0, 1])))

            item = QGraphicsPathItem(path)
            pen = QPen(color)
            pen.setWidthF(float(width))
            # apply alpha if provided
            if alpha is not None:
                try:
                    col = QColor(color)
                    col.setAlphaF(max(0.0, min(1.0, float(alpha))))
                    pen.setColor(col)
                except Exception:
                    pass
            # set dash pattern if requested (dash_len, dash_space)
            if dash_pattern is not None:
                try:
                    dash_len, dash_space = dash_pattern
                    pen.setDashPattern([float(dash_len), float(dash_space)])
                except Exception:
                    pass
            item.setPen(pen)
            item.setZValue(11_000)
            self.scene().addItem(item)
            self._overlays.append(item)
            self._register_overlay_item(item, tag)
        except Exception:
            logger.debug('Failed to add contour overlay', exc_info=True)

    def set_image(self, img: Union[QPixmap, QImage, np.ndarray, None], preserve_overlays: bool = False) -> None:
        if img is None:
            self._scene.clear()
            self._pix_item = None
            self._last_pm_size = None
            self._overlays.clear()
            self._overlay_items_by_tag.clear()
            self._tmp_item = None
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

        had_item = self._pix_item is not None
        same_size = had_item and (self._last_pm_size == QSizeF(pm.size()))
        if preserve_overlays and had_item and same_size:
            self._pix_item.setPixmap(pm)
            self._scene.setSceneRect(QRectF(pm.rect()))
            self._last_pm_size = QSizeF(pm.size())
            return

        # Preserve current transform/center if policy asks and size unchanged
        old_transform = QTransform(self.transform())
        old_center = self.mapToScene(self.viewport().rect().center())

        self._scene.clear()
        self._overlays.clear()
        self._overlay_items_by_tag.clear()
        self._tmp_item = None
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

    def _remove_overlay_item(self, item) -> None:
        try:
            scene = item.scene()
        except RuntimeError:
            scene = None
        if scene:
            try:
                scene.removeItem(item)
            except Exception:
                pass
        if item in self._overlays:
            self._overlays.remove(item)
        for key, value in list(self._overlay_items_by_tag.items()):
            if value is item:
                del self._overlay_items_by_tag[key]

    def _register_overlay_item(self, item, tag: str | None = None) -> None:
        tag = tag or self._draw_tag
        if not tag:
            return
        existing = self._overlay_items_by_tag.get(tag)
        if existing and existing is not item:
            self._remove_overlay_item(existing)
        self._overlay_items_by_tag[tag] = item

    def overlay_rect_scene(self, tag: str) -> QRectF | None:
        item = self._overlay_items_by_tag.get(tag)
        if isinstance(item, QGraphicsRectItem):
            rect = item.rect().normalized()
            top_left = item.mapToScene(rect.topLeft())
            bottom_right = item.mapToScene(rect.bottomRight())
            return QRectF(top_left, bottom_right)
        return None

    def overlay_line_scene(self, tag: str) -> QLineF | None:
        item = self._overlay_items_by_tag.get(tag)
        if isinstance(item, QGraphicsLineItem):
            line = item.line()
            p1 = item.mapToScene(line.p1())
            p2 = item.mapToScene(line.p2())
            return QLineF(p1, p2)
        return None

    def has_overlay(self, tag: str) -> bool:
        return tag in self._overlay_items_by_tag

    def _clamp_to_scene(self, point: QPointF) -> QPointF:
        if not self._scene:
            return point
        rect = self._scene.sceneRect()
        if rect.isNull():
            return point
        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())
        return QPointF(x, y)

    # ---------------- Mouse handling for drawing ----------------
    def mousePressEvent(self, event):
        scene_pos = None
        if event.button() == Qt.LeftButton and self.scene():
            candidate = self.mapToScene(event.pos())
            rect = self._scene.sceneRect() if self._scene else None
            if rect and rect.isValid():
                padded = rect.adjusted(-1e-6, -1e-6, 1e-6, 1e-6)
                if padded.contains(candidate):
                    scene_pos = self._clamp_to_scene(candidate)
                    self.point_clicked.emit(scene_pos, _enum_to_int(event.button()), _enum_to_int(event.modifiers()))
                else:
                    scene_pos = None
            else:
                scene_pos = self._clamp_to_scene(candidate)
                self.point_clicked.emit(scene_pos, _enum_to_int(event.button()), _enum_to_int(event.modifiers()))

        if self._draw_mode and event.button() == Qt.LeftButton and self.scene():
            if scene_pos is None:
                scene_pos = self.mapToScene(event.pos())
                rect = self._scene.sceneRect() if self._scene else None
                if rect and rect.isValid():
                    padded = rect.adjusted(-1e-6, -1e-6, 1e-6, 1e-6)
                    if not padded.contains(scene_pos):
                        return super().mousePressEvent(event)
                scene_pos = self._clamp_to_scene(scene_pos)
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

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.scene():
            candidate = self.mapToScene(event.pos())
            rect = self._scene.sceneRect() if self._scene else None
            if rect and rect.isValid():
                padded = rect.adjusted(-1e-6, -1e-6, 1e-6, 1e-6)
                if not padded.contains(candidate):
                    return super().mouseDoubleClickEvent(event)
            scene_pos = self._clamp_to_scene(candidate)
            self.double_clicked.emit(scene_pos, _enum_to_int(event.button()), _enum_to_int(event.modifiers()))
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self._draw_mode and self._tmp_item and self._press_pos_scene and self.scene():
            current_pos = self._clamp_to_scene(self.mapToScene(event.pos()))
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
                    rect = self._tmp_item.rect().normalized()
                    top_left = self._tmp_item.mapToScene(rect.topLeft())
                    bottom_right = self._tmp_item.mapToScene(rect.bottomRight())
                    tag = (self._draw_tag or 'region').lower()
                    label = 'ROI region' if tag == 'roi' else ('needle region' if tag == 'needle' else 'region')
                    logger.info("Preview %s drawn: top_left=(%.2f, %.2f), bottom_right=(%.2f, %.2f)", label, top_left.x(), top_left.y(), bottom_right.x(), bottom_right.y())
                    self.roi_selected.emit(self._tmp_item.rect())
                elif self._draw_mode == DRAW_LINE and isinstance(self._tmp_item, QGraphicsLineItem):
                    line = self._tmp_item.line()
                    p1 = self._tmp_item.mapToScene(line.p1())
                    p2 = self._tmp_item.mapToScene(line.p2())
                    tag = (self._draw_tag or 'contact_line').lower()
                    label = 'needle line' if tag == 'needle' else 'contact line'
                    logger.info("Preview %s drawn: p1=(%.2f, %.2f), p2=(%.2f, %.2f)", label, p1.x(), p1.y(), p2.x(), p2.y())
                    self.line_drawn.emit(self._tmp_item.line())

                # finalize
                item = self._tmp_item
                self._overlays.append(item)
                self._register_overlay_item(item)
                self._tmp_item = None
                self._press_pos_scene = None
            return
        super().mouseReleaseEvent(event)








