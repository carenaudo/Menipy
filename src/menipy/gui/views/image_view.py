"""Custom image viewer widget."""

# type: ignore
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
from typing import Optional, Union, Any
import numpy as np

from PySide6.QtCore import Qt, QRectF, QPointF, QSizeF, Signal, QLineF, QTimer
from PySide6.QtGui import QImage, QPixmap, QTransform, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsPathItem,
    QGraphicsSimpleTextItem,
)

# Drawing modes for overlays
DRAW_NONE = None
DRAW_POINT = "point"
DRAW_LINE = "line"
DRAW_RECT = "rect"

_TAG_LAYER_DEFAULTS = {
    "roi": "markers",
    "needle": "markers",
    "contact_line": "baseline",
    "detected_contour": "contour",
    "cal_drop": "contour",
    "result_contour": "contour",
    "pendant_fit": "fit",
    "axis": "axes",
    "apex": "markers",
}


def _enum_to_int(value: object) -> int:
    """Convert Qt enum/flag objects to plain ints for signal payloads."""
    if hasattr(value, "value"):
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
        self._fit_resize_pending: bool = False

        self._draw_mode = DRAW_NONE
        self._overlays = []  # list[QGraphicsItem]
        self._overlay_items_by_tag: dict[str, object] = {}
        self._overlay_layers_by_tag: dict[str, str] = {}
        self._overlay_layer_visibility: dict[str, bool] = {}
        self._overlay_stroke_scale_mode: str = "screen"
        self._marker_config: dict[str, object] = {}
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
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    # --------------------- public API ---------------------
    def set_auto_policy(self, policy: str = "auto") -> None:
        """'auto' (default): auto-fit/actual on new image; 'preserve': keep current pan/zoom if size unchanged."""

        if policy not in ("auto", "preserve"):
            raise ValueError("policy must be 'auto' or 'preserve'")
        self._auto_policy = policy

    def set_wheel_zoom_requires_ctrl(self, required: bool) -> None:
        self._wheel_zoom_requires_ctrl = bool(required)

    def set_draw_mode(
        self,
        mode: str | None,
        color: QColor = QColor(255, 0, 0),
        tag: str | None = None,
    ) -> None:
        """Update active drawing mode and overlay color."""

        self._draw_mode = mode
        self._draw_tag = tag if mode else None
        self._overlay_pen.setColor(color)
        self._configure_pen(self._overlay_pen)
        # When drawing, disable scroll-hand drag for comfort
        self.setDragMode(QGraphicsView.NoDrag if mode else QGraphicsView.ScrollHandDrag)

    def clear_overlays(self) -> None:
        """clear_overlays."""
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
        self._overlay_layers_by_tag.clear()
        self._tmp_item = None

    def remove_overlay(self, tag: str) -> None:
        item = self._overlay_items_by_tag.get(tag)
        if not item:
            return
        self._remove_overlay_item(item)
        self._overlay_layers_by_tag.pop(tag, None)

    def set_overlay_layer_visible(self, layer: str, visible: bool) -> None:
        """Show or hide all existing overlay items assigned to a logical layer."""
        layer = str(layer or "").strip()
        if not layer:
            return
        self._overlay_layer_visibility[layer] = bool(visible)
        for item in list(self._overlays):
            try:
                item_layer = item.data(1)
            except Exception:
                item_layer = None
            if item_layer == layer:
                try:
                    item.setVisible(bool(visible))
                except RuntimeError:
                    pass

    def overlay_layer_visible(self, layer: str) -> bool:
        return bool(self._overlay_layer_visibility.get(layer, True))

    def set_overlay_stroke_scale_mode(self, mode: str) -> None:
        self._overlay_stroke_scale_mode = (
            "image" if str(mode).lower() == "image" else "screen"
        )

    def set_marker_config(self, config: dict | None) -> None:
        self._marker_config = dict(config or {})

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
        layer: str | None = None,
        tag: str | None = None,
    ) -> None:
        # Create shape-specific QGraphicsItem
        layer = layer or self._infer_overlay_layer(tag)
        marker_style = self._marker_style_for_tag(tag) if layer == "markers" else {}
        if marker_style.get("visible") is False:
            return
        if "color" in marker_style:
            color = QColor(str(marker_style["color"]))
        if "radius" in marker_style:
            try:
                radius = float(marker_style["radius"])
            except Exception:
                pass
        if "shape" in marker_style:
            shape = str(marker_style["shape"])
        if shape == "square":
            item = QGraphicsRectItem(
                QRectF(point.x() - radius, point.y() - radius, 2 * radius, 2 * radius)
            )
        elif shape == "cross":
            from PySide6.QtGui import QPainterPath

            path = QPainterPath()
            path.moveTo(point.x() - radius, point.y() - radius)
            path.lineTo(point.x() + radius, point.y() + radius)
            path.moveTo(point.x() - radius, point.y() + radius)
            path.lineTo(point.x() + radius, point.y() - radius)
            item = QGraphicsPathItem(path)
            pen = QPen(color)
            pen.setWidthF(float(stroke_width))
            self._configure_pen(pen)
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
            item.setPen(pen)
            item.setZValue(12_000)
            self.scene().addItem(item)
            self._overlays.append(item)
            self._register_overlay_item(item, tag, layer)
            self._maybe_add_marker_label(QPointF(point), tag, marker_style)
            return
        else:
            item = QGraphicsEllipseItem(
                QRectF(point.x() - radius, point.y() - radius, 2 * radius, 2 * radius)
            )

        pen = QPen(color)
        pen.setWidthF(float(stroke_width))
        self._configure_pen(pen)
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
        self._register_overlay_item(item, tag, layer)
        self._maybe_add_marker_label(QPointF(point), tag, marker_style)

    def add_marker_line(
        self,
        p1: QPointF,
        p2: QPointF,
        *,
        color: QColor = QColor(255, 140, 0),
        width: float = 2.0,
        layer: str | None = None,
        tag: str | None = None,
    ) -> None:
        item = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
        pen = QPen(color)
        pen.setWidthF(float(width))
        self._configure_pen(pen)
        item.setPen(pen)
        item.setZValue(11_000)
        self.scene().addItem(item)
        self._overlays.append(item)
        self._register_overlay_item(item, tag, layer)

    def add_marker_rect(
        self,
        rect: QRectF,
        *,
        color: QColor = QColor(255, 255, 0),
        width: float = 2.0,
        layer: str | None = None,
        tag: str | None = None,
    ) -> None:
        """Add a rectangle overlay.

        Args:
            rect: Rectangle in scene coordinates (can be QRectF or tuple of x,y,w,h)
            color: Outline color
            width: Line width
            tag: Optional tag for later retrieval/removal
        """
        if isinstance(rect, tuple):
            x, y, w, h = rect
            rect = QRectF(x, y, w, h)

        layer = layer or self._infer_overlay_layer(tag)
        marker_style = self._marker_style_for_tag(tag) if layer == "markers" else {}
        if marker_style.get("visible") is False:
            return
        if "color" in marker_style:
            color = QColor(str(marker_style["color"]))

        item = QGraphicsRectItem(rect)
        pen = QPen(color)
        pen.setWidthF(float(width))
        self._configure_pen(pen)
        item.setPen(pen)
        item.setZValue(11_000)
        self.scene().addItem(item)
        self._overlays.append(item)
        self._register_overlay_item(item, tag, layer)
        self._maybe_add_marker_label(rect.center(), tag, marker_style)

    def add_marker_contour(
        self,
        pts: np.ndarray,
        *,
        color: QColor = QColor(255, 0, 0),
        width: float = 2.0,
        dash_pattern: tuple[float, float] | None = None,
        alpha: float | None = None,
        closed: bool = True,
        layer: str | None = None,
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
            if closed:
                path.lineTo(QPointF(float(pts[0, 0]), float(pts[0, 1])))

            item = QGraphicsPathItem(path)
            pen = QPen(color)
            pen.setWidthF(float(width))
            self._configure_pen(pen)
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
            self._register_overlay_item(item, tag, layer)
        except Exception:
            logger.debug("Failed to add contour overlay", exc_info=True)

    def add_marker_text(
        self,
        point: QPointF,
        text: str,
        *,
        color: QColor = QColor(255, 255, 255),
        scale: float = 1.0,
        layer: str | None = None,
        tag: str | None = None,
    ) -> None:
        layer = layer or self._infer_overlay_layer(tag)
        marker_style = self._marker_style_for_tag(tag) if layer == "markers" else {}
        if marker_style.get("visible") is False:
            return
        if marker_style:
            color = QColor(
                str(
                    marker_style.get(
                        "label_color", marker_style.get("color", color.name())
                    )
                )
            )
        item = QGraphicsSimpleTextItem(str(text))
        item.setPos(point)
        item.setBrush(color)
        try:
            font = item.font()
            if marker_style.get("font_family"):
                font.setFamily(str(marker_style["font_family"]))
            if marker_style.get("font_size"):
                font.setPointSizeF(float(marker_style["font_size"]))
            item.setFont(font)
            item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
        except Exception:
            pass
        try:
            item.setScale(float(scale))
        except Exception:
            pass
        item.setZValue(13_000)
        self.scene().addItem(item)
        self._overlays.append(item)
        self._register_overlay_item(item, tag, layer)

    def _configure_pen(self, pen: QPen) -> None:
        try:
            pen.setCosmetic(self._overlay_stroke_scale_mode != "image")
        except Exception:
            pass

    def _marker_style_for_tag(self, tag: str | None) -> dict[str, object]:
        config = self._marker_config or {}
        marker_type = self._marker_type_for_tag(tag)
        style = (
            dict(config.get("default", {}))
            if isinstance(config.get("default"), dict)
            else {}
        )
        specific = config.get(marker_type)
        if isinstance(specific, dict):
            style.update(specific)
        return style

    def _marker_type_for_tag(self, tag: str | None) -> str:
        tag = tag or ""
        if "roi" in tag:
            return "roi"
        if "needle" in tag:
            return "needle"
        if "contact" in tag or "anchor" in tag:
            return "contact"
        if "apex" in tag:
            return "apex"
        if "center" in tag:
            return "drop_center"
        if "bg" in tag or "background" in tag:
            return "background"
        if "text" in tag:
            return "result_text"
        return "default"

    def _maybe_add_marker_label(
        self, point: QPointF, tag: str | None, style: dict[str, object]
    ) -> None:
        if not style.get("label_visible"):
            return
        label = str(style.get("label_text") or self._marker_type_for_tag(tag))
        if not label:
            return
        item = QGraphicsSimpleTextItem(label)
        item.setPos(point + QPointF(6, -18))
        item.setBrush(
            QColor(str(style.get("label_color", style.get("color", "white"))))
        )
        try:
            font = item.font()
            if style.get("font_family"):
                font.setFamily(str(style["font_family"]))
            if style.get("font_size"):
                font.setPointSizeF(float(style["font_size"]))
            item.setFont(font)
            item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
        except Exception:
            pass
        item.setZValue(13_000)
        self.scene().addItem(item)
        self._overlays.append(item)
        self._register_overlay_item(item, f"{tag}_label" if tag else None, "markers")

    def set_image(
        self,
        img: Union[QPixmap, QImage, np.ndarray, None],
        preserve_overlays: bool = False,
    ) -> None:
        if img is None:
            self._scene.clear()
            self._pix_item = None
            self._last_pm_size = None
            self._overlays.clear()
            self._overlay_items_by_tag.clear()
            self._overlay_layers_by_tag.clear()
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
        self._overlay_layers_by_tag.clear()
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
        """actual_size."""
        if not self._pix_item:
            return
        self.setTransform(QTransform())  # identity
        self.centerOn(self._pix_item)
        self._mode = "actual"

    def fit_to_window(self) -> None:
        """fit_to_window."""
        if not self._pix_item:
            return
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._mode = "fit"

        # --------------------- events & helpers ---------------------

    def wheelEvent(self, event):
        """wheel event."""
        # Require Ctrl only if configured
        if self._wheel_zoom_requires_ctrl and not (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            return super().wheelEvent(event)
        # Otherwise zoom w/o modifier
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        event.accept()

    def resizeEvent(self, event):
        """resize event."""
        super().resizeEvent(event)
        if getattr(self, "_mode", None) == "fit":
            self._schedule_fit_to_window()

    def _schedule_fit_to_window(self) -> None:
        if self._fit_resize_pending:
            return
        self._fit_resize_pending = True
        QTimer.singleShot(0, self._run_scheduled_fit)

    def _run_scheduled_fit(self) -> None:
        self._fit_resize_pending = False
        if getattr(self, "_mode", None) == "fit" and self._pix_item:
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
        """_remove_overlay_item."""
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

    def _set_overlay_item_metadata(
        self, item: Any, tag: str | None = None, layer: str | None = None
    ) -> None:
        tag = tag or self._draw_tag
        layer = layer or (self._infer_overlay_layer(tag) if tag else None)
        if tag:
            try:
                item.setData(0, tag)
            except Exception:
                pass
        if layer:
            try:
                item.setData(1, layer)
                item.setVisible(self.overlay_layer_visible(layer))
            except Exception:
                pass

    def _register_overlay_item(
        self, item, tag: str | None = None, layer: str | None = None
    ) -> None:
        tag = tag or self._draw_tag
        layer = layer or (self._infer_overlay_layer(tag) if tag else None)
        self._set_overlay_item_metadata(item, tag, layer)
        if not tag:
            return
        existing = self._overlay_items_by_tag.get(tag)
        if existing and existing is not item:
            self._remove_overlay_item(existing)
        self._overlay_items_by_tag[tag] = item
        if layer:
            self._overlay_layers_by_tag[tag] = layer

    def _infer_overlay_layer(self, tag: str | None) -> str | None:
        if not tag:
            return None
        if tag in _TAG_LAYER_DEFAULTS:
            return _TAG_LAYER_DEFAULTS[tag]
        if tag.startswith("marker_") or tag.startswith("cal_contact"):
            return "markers"
        return None

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
        """Mouse press event."""
        scene_pos = None
        if event.button() == Qt.LeftButton and self.scene():
            candidate = self.mapToScene(event.pos())
            rect = self._scene.sceneRect() if self._scene else None
            if rect and rect.isValid():
                padded = rect.adjusted(-1e-6, -1e-6, 1e-6, 1e-6)
                if padded.contains(candidate):
                    scene_pos = self._clamp_to_scene(candidate)
                    self.point_clicked.emit(
                        scene_pos,
                        _enum_to_int(event.button()),
                        _enum_to_int(event.modifiers()),
                    )
                else:
                    scene_pos = None
            else:
                scene_pos = self._clamp_to_scene(candidate)
                self.point_clicked.emit(
                    scene_pos,
                    _enum_to_int(event.button()),
                    _enum_to_int(event.modifiers()),
                )

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
                    QRectF(
                        scene_pos.x() - radius,
                        scene_pos.y() - radius,
                        2 * radius,
                        2 * radius,
                    )
                )
                item.setPen(self._overlay_pen)
                item.setZValue(10_000)
                self.scene().addItem(item)
                self._overlays.append(item)
                self._register_overlay_item(item)
                self._press_pos_scene = None
            elif self._draw_mode == DRAW_LINE:
                item = QGraphicsLineItem(
                    scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y()
                )
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
        """Mouse double click event."""
        if event.button() == Qt.LeftButton and self.scene():
            candidate = self.mapToScene(event.pos())
            rect = self._scene.sceneRect() if self._scene else None
            if rect and rect.isValid():
                padded = rect.adjusted(-1e-6, -1e-6, 1e-6, 1e-6)
                if not padded.contains(candidate):
                    return super().mouseDoubleClickEvent(event)
            scene_pos = self._clamp_to_scene(candidate)
            self.double_clicked.emit(
                scene_pos, _enum_to_int(event.button()), _enum_to_int(event.modifiers())
            )
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        """Mouse move event."""
        if (
            self._draw_mode
            and self._tmp_item
            and self._press_pos_scene
            and self.scene()
        ):
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
        """mouseReleaseEvent.

        Parameters
        ----------
        event : type
        Description.
        """
        if self._draw_mode and event.button() == Qt.LeftButton and self.scene():
            if self._tmp_item:
                if self._draw_mode == DRAW_RECT and isinstance(
                    self._tmp_item, QGraphicsRectItem
                ):
                    rect = self._tmp_item.rect().normalized()
                    top_left = self._tmp_item.mapToScene(rect.topLeft())
                    bottom_right = self._tmp_item.mapToScene(rect.bottomRight())
                    tag = (self._draw_tag or "region").lower()
                    label = (
                        "ROI region"
                        if tag == "roi"
                        else ("needle region" if tag == "needle" else "region")
                    )
                    logger.info(
                        "Preview %s drawn: top_left=(%.2f, %.2f), bottom_right=(%.2f, %.2f)",
                        label,
                        top_left.x(),
                        top_left.y(),
                        bottom_right.x(),
                        bottom_right.y(),
                    )
                    self.roi_selected.emit(self._tmp_item.rect())
                elif self._draw_mode == DRAW_LINE and isinstance(
                    self._tmp_item, QGraphicsLineItem
                ):
                    line = self._tmp_item.line()
                    p1 = self._tmp_item.mapToScene(line.p1())
                    p2 = self._tmp_item.mapToScene(line.p2())
                    tag = (self._draw_tag or "contact_line").lower()
                    label = "needle line" if tag == "needle" else "contact line"
                    logger.info(
                        "Preview %s drawn: p1=(%.2f, %.2f), p2=(%.2f, %.2f)",
                        label,
                        p1.x(),
                        p1.y(),
                        p2.x(),
                        p2.y(),
                    )
                    self.line_drawn.emit(self._tmp_item.line())

                # finalize
                item = self._tmp_item
                self._overlays.append(item)
                self._register_overlay_item(item)
                self._tmp_item = None
                self._press_pos_scene = None
            return
        super().mouseReleaseEvent(event)
