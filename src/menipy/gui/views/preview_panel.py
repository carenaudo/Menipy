"""Live preview panel for image display and interaction."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QWidget,
)

from menipy.gui.helpers.icon_loader import load_icon, set_button_icon

LAYER_DEFAULTS = {
    "contour": True,
    "fit": True,
    "axes": True,
    "baseline": True,
    "markers": True,
}

LAYER_SETTINGS_KEYS = {
    "contour": "contour_visible",
    "fit": "fit_visible",
    "axes": "axes_visible",
    "baseline": "baseline_visible",
    "markers": "markers_visible",
}

LAYER_STYLE_KEYS = {
    "contour": ("contour_color", "contour_thickness", "contour_alpha"),
    "fit": ("fit_color", "fit_thickness", "fit_alpha"),
    "axes": ("axis_color", "axis_thickness", "axis_alpha"),
    "baseline": ("baseline_color", "baseline_thickness", "baseline_alpha"),
    "markers": ("marker_color", "marker_thickness", "marker_alpha"),
}

TAG_LAYER_OVERRIDES = {
    "detected_contour": "contour",
    "cal_drop": "contour",
    "result_contour": "contour",
    "pendant_fit": "fit",
    "axis": "axes",
    "symmetry_axis": "axes",
    "baseline": "baseline",
    "contact_line": "baseline",
    "roi": "markers",
    "needle": "markers",
    "cal_needle": "markers",
    "cal_roi": "markers",
    "cal_contact_left": "markers",
    "cal_contact_right": "markers",
    "apex": "markers",
    "measurement_text": "markers",
    "fit_text": "markers",
}


class PreviewPanel:
    """Encapsulates the overlay/preview area interactions."""

    roi_selected = Signal(QRectF)
    line_drawn = Signal(QLineF)

    def __init__(
        self,
        panel: QWidget,
        image_view_cls: Optional[type[Any]],
        settings: Optional[Any] = None,
    ) -> None:
        self.panel = panel
        self.settings = settings
        self.image_view = getattr(panel, "previewView", None)
        if not self.image_view and image_view_cls is not None:
            self.image_view = panel.findChild(image_view_cls, "previewView")

        self._on_roi_selected: Optional[Callable[[QRectF], None]] = None
        self._on_line_drawn: Optional[Callable[[QLineF], None]] = None
        self._overlay_buttons: list[QToolButton | QPushButton] = []
        self._layer_actions: dict[str, QAction] = {}
        self._layer_checks: dict[str, QCheckBox] = {}
        self._layer_state = self._load_layer_state()
        self._status_label: QLabel | None = self.panel.findChild(
            QLabel, "previewStatusLabel"
        )
        if self._status_label:
            self._status_label.setStyleSheet(
                "color: #57606A; font-weight: 600; padding: 2px 8px;"
            )

        if self.image_view:
            self._configure_image_view()
            self.image_view.roi_selected.connect(self.on_roi_selected)
            self.image_view.line_drawn.connect(self.on_line_drawn)

        self._install_guided_menus()
        self._apply_control_icons()
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
                self._set_status(f"File · {target.name}")
                return
            except Exception as exc:
                raise RuntimeError(f"Failed to load image {target}: {exc}") from exc
        if hasattr(self.image_view, "setImage"):
            self.image_view.setImage(str(target))
            logger.info("Preview loaded from %s", target)
            self._set_overlay_buttons_enabled(True)
            self._set_status(f"File · {target.name}")
            return
        if hasattr(self.image_view, "load"):
            self.image_view.load(str(target))
            logger.info("Preview loaded from %s", target)
            self._set_overlay_buttons_enabled(True)
            self._set_status(f"File · {target.name}")
            return
        raise RuntimeError("Preview ImageView has no loader method")

    def display(self, payload: Any) -> None:
        if not self.image_view:
            return
        if payload is None:
            return
        if isinstance(payload, (str, Path)):
            self.load_path(payload)
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

    def display_context(self, ctx: Any) -> None:
        """Display a completed pipeline context with interactive overlay layers."""
        if not self.image_view or ctx is None:
            return
        base = self._context_base_image(ctx)
        commands = list(getattr(ctx, "overlay_commands", None) or [])
        if base is not None and commands:
            self.display(base)
            self.render_overlay_commands(commands)
            self._set_status("Analysis preview · overlays")
            return
        if getattr(ctx, "preview", None) is not None:
            self.display(ctx.preview)
            self._set_status("Analysis preview")
            return
        if base is not None:
            self.display(base)
            self._set_status("Analysis preview")

    def render_overlay_commands(self, commands: list[dict[str, Any]]) -> None:
        if not self.image_view:
            return
        self._clear_result_overlays()
        for index, cmd in enumerate(commands):
            if not isinstance(cmd, dict):
                continue
            self._render_overlay_command(cmd, index)
        self._apply_all_layer_visibility()

    def set_draw_mode(
        self, mode: Any, color: QColor = QColor(255, 0, 0), *, tag: Optional[str] = None
    ) -> None:
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

    def apply_overlay_config(self, config: dict[str, Any] | None = None) -> None:
        cfg = (
            config
            if config is not None
            else getattr(self.settings, "overlay_config", None)
        )
        cfg = dict(cfg or {})
        self._layer_state = self._load_layer_state(cfg)
        if self.image_view and hasattr(
            self.image_view, "set_overlay_stroke_scale_mode"
        ):
            self.image_view.set_overlay_stroke_scale_mode(
                str(cfg.get("stroke_scale_mode", "screen"))
            )
        for layer, visible in self._layer_state.items():
            check = self._layer_checks.get(layer)
            action = self._layer_actions.get(layer)
            if check:
                check.blockSignals(True)
                check.setChecked(bool(visible))
                check.blockSignals(False)
            if action:
                action.blockSignals(True)
                action.setChecked(bool(visible))
                action.blockSignals(False)
        self._apply_all_layer_visibility()

    def apply_marker_config(self, config: dict[str, Any] | None = None) -> None:
        if self.image_view and hasattr(self.image_view, "set_marker_config"):
            self.image_view.set_marker_config(config or {})

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
        """has_roi."""
        return self.roi_rect() is not None

    def has_needle(self) -> bool:
        """has_needle."""
        return self.needle_rect() is not None

    def has_contact_line(self) -> bool:
        """has_contact_line."""
        return self.contact_line_segment() is not None

    def roi_rect(self) -> tuple[int, int, int, int] | None:
        """roi_rect."""
        rect = self._overlay_rect("roi")
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
        """needle_rect."""
        rect = self._overlay_rect("needle")
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
        """contact line segment.

        Returns
        -------
        type
        Description.
        """
        line = self._overlay_line("contact_line")
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
        if not self.image_view or not hasattr(self.image_view, "overlay_rect_scene"):
            return None
        return self.image_view.overlay_rect_scene(tag)

    def _overlay_line(self, tag: str) -> QLineF | None:
        if not self.image_view or not hasattr(self.image_view, "overlay_line_scene"):
            return None
        return self.image_view.overlay_line_scene(tag)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _configure_image_view(self) -> None:
        """_configure_image_view."""
        try:
            self.image_view.set_auto_policy("preserve")
        except Exception:
            pass
        if hasattr(self.image_view, "set_overlay_layer_visible"):
            for layer, visible in self._layer_state.items():
                self.image_view.set_overlay_layer_visible(layer, visible)
        cfg = getattr(self.settings, "overlay_config", None) or {}
        if hasattr(self.image_view, "set_overlay_stroke_scale_mode"):
            self.image_view.set_overlay_stroke_scale_mode(
                str(cfg.get("stroke_scale_mode", "screen"))
            )
        if hasattr(self.image_view, "set_marker_config"):
            self.image_view.set_marker_config(
                getattr(self.settings, "marker_config", {}) or {}
            )
        try:
            self.image_view.set_wheel_zoom_requires_ctrl(False)
        except Exception:
            pass

    def _wire_buttons(self) -> None:
        """_wire_buttons."""

        def _find_button(name: str) -> Optional[QToolButton | QPushButton]:
            button = self.panel.findChild(QToolButton, name)
            if button:
                return button
            return self.panel.findChild(QPushButton, name)

        if not self.image_view:
            return

        try:
            from menipy.gui.views.image_view import DRAW_LINE, DRAW_RECT
        except ImportError:
            return

        actions = {
            "roiBtn": lambda: self.set_draw_mode(
                DRAW_RECT, QColor(255, 255, 0), tag="roi"
            ),  # Yellow
            "needleBtn": lambda: self.set_draw_mode(
                DRAW_RECT, QColor(0, 0, 255), tag="needle"
            ),  # Blue
            "contactLineBtn": lambda: self.set_draw_mode(
                DRAW_LINE, QColor(173, 216, 230), tag="contact_line"
            ),  # Light Blue
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

    def _apply_control_icons(self) -> None:
        """Apply resource-backed icons to preview and transport controls."""

        def _find_button(name: str) -> Optional[QToolButton | QPushButton]:
            button = self.panel.findChild(QToolButton, name)
            if button:
                return button
            return self.panel.findChild(QPushButton, name)

        for button_name, icon_name in (
            ("overlayMenuBtn", "overlay"),
            ("markMenuBtn", "roi"),
            ("actualBtn", "zoom-in"),
            ("fitBtn", "zoom-out"),
            ("roiBtn", "roi"),
            ("needleBtn", "pendant"),
            ("contactLineBtn", "sessile"),
            ("clearBtn", "x"),
        ):
            set_button_icon(_find_button(button_name), icon_name, size=15)

        for button_name, icon_name in (
            ("frameBackBtn", "skip-back"),
            ("framePlayBtn", "play"),
            ("frameForwardBtn", "skip-forward"),
        ):
            button = _find_button(button_name)
            set_button_icon(button, icon_name, size=16, clear_text=True)
            if isinstance(button, QToolButton):
                button.setToolButtonStyle(Qt.ToolButtonIconOnly)

    def _set_status(self, text: str) -> None:
        if self._status_label:
            self._status_label.setText(text)

    def _install_guided_menus(self) -> None:
        """Replace exposed overlay controls with compact guided menus."""

        def _find_button(name: str) -> Optional[QToolButton | QPushButton]:
            button = self.panel.findChild(QToolButton, name)
            if button:
                return button
            return self.panel.findChild(QPushButton, name)

        toggles_layout = self.panel.findChild(QHBoxLayout, "togglesLayout")
        mark_layout = self.panel.findChild(QHBoxLayout, "horizontalLayout")

        overlay_button = self.panel.findChild(QToolButton, "overlayMenuBtn")
        if overlay_button is None:
            overlay_button = QToolButton(self.panel)
            overlay_button.setObjectName("overlayMenuBtn")
            overlay_button.setText("Overlays")
            overlay_button.setPopupMode(QToolButton.InstantPopup)
            if hasattr(toggles_layout, "insertWidget"):
                toggles_layout.insertWidget(0, overlay_button)

        if overlay_button:
            overlay_button.setVisible(False)

        overlay_menu = QMenu(overlay_button)
        layer_defs = (
            ("contour", "showContourCheck", "Contour"),
            ("fit", "showFitCheck", "Fit"),
            ("axes", "showAxesCheck", "Axis"),
            ("baseline", "showBaselineCheck", "Baseline"),
            ("markers", "showMarkersCheck", "Markers"),
        )
        insert_index = 0
        for layer, checkbox_name, label in layer_defs:
            checkbox = self.panel.findChild(QCheckBox, checkbox_name)
            if not checkbox:
                checkbox = QCheckBox(self.panel)
                checkbox.setObjectName(checkbox_name)
                if toggles_layout and hasattr(toggles_layout, "insertWidget"):
                    toggles_layout.insertWidget(insert_index, checkbox)
                    insert_index += 1
            action = QAction(label, overlay_menu)
            action.setCheckable(True)
            checked = bool(self._layer_state.get(layer, True))
            checkbox.setText(label)
            checkbox.setStyleSheet("""
                QCheckBox {
                    border: 1px solid #D0D7DE;
                    border-radius: 4px;
                    padding: 4px 8px;
                    background: #FFFFFF;
                    color: #57606A;
                    font-weight: 600;
                }
                QCheckBox:hover {
                    background: #EAEFF4;
                    color: #24292F;
                }
                QCheckBox:checked {
                    background: #DDF4FF;
                    border-color: #54AEFF;
                    color: #0969DA;
                }
                QCheckBox::indicator {
                    width: 0px;
                    height: 0px;
                }
            """)
            checkbox.setChecked(checked)
            action.setChecked(checked)
            action.toggled.connect(checkbox.setChecked)
            checkbox.toggled.connect(action.setChecked)
            checkbox.toggled.connect(
                lambda checked=False, layer_name=layer: self.set_layer_visible(
                    layer_name, checked
                )
            )
            overlay_menu.addAction(action)
            checkbox.setVisible(True)
            self._layer_actions[layer] = action
            self._layer_checks[layer] = checkbox
        overlay_button.setMenu(overlay_menu)

        mark_button = self.panel.findChild(QToolButton, "markMenuBtn")
        if mark_button is None:
            mark_button = QToolButton(self.panel)
            mark_button.setObjectName("markMenuBtn")
            mark_button.setText("Mark")
            mark_button.setPopupMode(QToolButton.InstantPopup)
            if hasattr(mark_layout, "insertWidget"):
                mark_layout.insertWidget(0, mark_button)

        mark_menu = QMenu(mark_button)
        for button_name, label, icon_name in (
            ("roiBtn", "ROI", "roi"),
            ("needleBtn", "Needle", "pendant"),
            ("contactLineBtn", "Contact Line", "sessile"),
            ("clearBtn", "Clear", "x"),
        ):
            button = _find_button(button_name)
            if not button:
                continue
            action = QAction(label, mark_menu)
            icon = load_icon(icon_name)
            if not icon.isNull():
                action.setIcon(icon)
            action.triggered.connect(button.click)
            mark_menu.addAction(action)
            button.setVisible(False)
        mark_button.setMenu(mark_menu)

    def set_layer_visible(self, layer: str, visible: bool) -> None:
        self._layer_state[layer] = bool(visible)
        if self.image_view and hasattr(self.image_view, "set_overlay_layer_visible"):
            self.image_view.set_overlay_layer_visible(layer, bool(visible))
        self._save_layer_state()

    def _load_layer_state(
        self, config: dict[str, Any] | None = None
    ) -> dict[str, bool]:
        cfg = (
            config
            if config is not None
            else getattr(self.settings, "overlay_config", None)
        )
        cfg = dict(cfg or {})
        return {
            layer: bool(cfg.get(LAYER_SETTINGS_KEYS[layer], default))
            for layer, default in LAYER_DEFAULTS.items()
        }

    def _save_layer_state(self) -> None:
        if self.settings is None:
            return
        cfg = dict(getattr(self.settings, "overlay_config", None) or {})
        for layer, key in LAYER_SETTINGS_KEYS.items():
            cfg[key] = bool(self._layer_state.get(layer, True))
        try:
            self.settings.overlay_config = cfg
            self.settings.save()
        except Exception:
            pass

    def _apply_all_layer_visibility(self) -> None:
        if not self.image_view or not hasattr(
            self.image_view, "set_overlay_layer_visible"
        ):
            return
        for layer, visible in self._layer_state.items():
            self.image_view.set_overlay_layer_visible(layer, visible)

    def _context_base_image(self, ctx: Any) -> Any:
        image = getattr(ctx, "image", None)
        if image is not None:
            return image
        frame = getattr(ctx, "current_frame", None)
        if frame is not None:
            data = getattr(frame, "data", None)
            if data is not None:
                return data
        frames = getattr(ctx, "frames", None)
        if frames is not None:
            try:
                if isinstance(frames, list) and frames:
                    first = frames[0]
                    return getattr(first, "data", first)
                if hasattr(frames, "ndim") and frames.ndim in (2, 3):
                    return frames
            except Exception:
                pass
        return None

    def _clear_result_overlays(self) -> None:
        if not self.image_view or not hasattr(self.image_view, "remove_overlay"):
            return
        for tag in (
            "result_contour",
            "pendant_fit",
            "axis",
            "apex",
            "measurement_text",
            "fit_text",
            "result_baseline",
        ):
            try:
                self.image_view.remove_overlay(tag)
            except Exception:
                pass

    def _render_overlay_command(self, cmd: dict[str, Any], index: int) -> None:
        if not self.image_view:
            return
        typ = cmd.get("type")
        tag = str(cmd.get("tag") or self._default_tag_for_command(cmd, index))
        layer = str(cmd.get("layer") or self._layer_for_tag(tag, typ))
        color = QColor(str(cmd.get("color", "white")))
        thickness = float(cmd.get("thickness", 2))
        alpha = cmd.get("alpha", None)
        color, thickness, alpha = self._style_for_layer(layer, color, thickness, alpha)
        try:
            if typ == "polyline":
                self.image_view.add_marker_contour(
                    cmd.get("points"),
                    color=color,
                    width=thickness,
                    alpha=alpha,
                    closed=bool(cmd.get("closed", True)),
                    tag=tag,
                    layer=layer,
                )
            elif typ == "line":
                p1 = cmd.get("p1")
                p2 = cmd.get("p2")
                if p1 is None or p2 is None:
                    return
                self.image_view.add_marker_line(
                    QPointF(float(p1[0]), float(p1[1])),
                    QPointF(float(p2[0]), float(p2[1])),
                    color=color,
                    width=thickness,
                    tag=tag,
                    layer=layer,
                )
            elif typ == "cross":
                p = cmd.get("p")
                if p is None:
                    return
                self.image_view.add_marker_point(
                    QPointF(float(p[0]), float(p[1])),
                    color=color,
                    radius=float(cmd.get("size", 6)),
                    alpha=alpha,
                    stroke_width=thickness,
                    shape="cross",
                    tag=tag,
                    layer=layer,
                )
            elif typ == "text":
                p = cmd.get("p")
                text = cmd.get("text", "")
                if p is None or not text:
                    return
                self.image_view.add_marker_text(
                    QPointF(float(p[0]), float(p[1])),
                    str(text),
                    color=color,
                    scale=float(cmd.get("scale", 1.0)),
                    tag=tag,
                    layer=layer,
                )
            elif typ == "circle":
                center = cmd.get("center")
                if center is None:
                    return
                self.image_view.add_marker_point(
                    QPointF(float(center[0]), float(center[1])),
                    color=color,
                    radius=float(cmd.get("radius", 4)),
                    alpha=alpha,
                    stroke_width=thickness,
                    tag=tag,
                    layer=layer,
                )
            elif typ == "scatter":
                points = cmd.get("points") or []
                for point_index, point in enumerate(points):
                    self.image_view.add_marker_point(
                        QPointF(float(point[0]), float(point[1])),
                        color=color,
                        radius=float(cmd.get("radius", 2)),
                        alpha=alpha,
                        tag=f"{tag}_{point_index}",
                        layer=layer,
                    )
        except Exception:
            logger.debug("Failed to render overlay command %s", typ, exc_info=True)

    def _style_for_layer(
        self,
        layer: str,
        color: QColor,
        thickness: float,
        alpha: Any,
    ) -> tuple[QColor, float, float | None]:
        cfg = getattr(self.settings, "overlay_config", None) or {}
        keys = LAYER_STYLE_KEYS.get(layer)
        if not keys:
            return color, thickness, float(alpha) if alpha is not None else None
        color_key, thickness_key, alpha_key = keys
        if color_key in cfg:
            color = QColor(str(cfg[color_key]))
        if thickness_key in cfg:
            try:
                thickness = float(cfg[thickness_key])
            except (TypeError, ValueError):
                pass
        if alpha is None:
            alpha = cfg.get(alpha_key)
        try:
            alpha_value = max(0.0, min(1.0, float(alpha)))
            color.setAlphaF(alpha_value)
            return color, thickness, alpha_value
        except (TypeError, ValueError):
            return color, thickness, None

    def _default_tag_for_command(self, cmd: dict[str, Any], index: int) -> str:
        typ = cmd.get("type")
        if typ == "polyline":
            return "result_contour"
        if typ == "line":
            return "axis"
        if typ == "cross":
            return "apex"
        if typ == "text":
            return "fit_text" if index else "measurement_text"
        return f"result_overlay_{index}"

    def _layer_for_tag(self, tag: str, typ: Any) -> str:
        if tag in TAG_LAYER_OVERRIDES:
            return TAG_LAYER_OVERRIDES[tag]
        if typ == "polyline":
            return "contour"
        if typ == "line":
            return "axes"
        if typ in {"cross", "circle", "scatter", "text"}:
            return "markers"
        return "markers"

    def _set_overlay_buttons_enabled(self, enabled: bool) -> None:
        if not self._overlay_buttons:
            return
        for button in self._overlay_buttons:
            try:
                button.setEnabled(enabled)
            except Exception:
                pass
