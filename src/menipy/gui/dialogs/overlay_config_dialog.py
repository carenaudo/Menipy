"""Dialog for overlay visualization settings."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class OverlayConfigDialog(QDialog):
    """Configure overlay layer appearance without changing public config keys."""

    previewRequested = Signal(object)
    configApplied = Signal(object)

    DEFAULTS = {
        "contour_visible": True,
        "contour_color": "red",
        "contour_thickness": 2,
        "contour_dashed": False,
        "contour_dash_length": 6,
        "contour_dash_space": 6,
        "contour_alpha": 0.9,
        "points_visible": True,
        "point_color": "lime",
        "point_radius": 6,
        "point_alpha": 0.95,
        "fit_visible": True,
        "fit_color": "#00aa00",
        "fit_thickness": 2,
        "fit_alpha": 0.95,
        "axes_visible": True,
        "axis_color": "#00ffff",
        "axis_thickness": 1,
        "axis_alpha": 0.85,
        "baseline_visible": True,
        "baseline_color": "#ff00ff",
        "baseline_thickness": 2,
        "baseline_alpha": 0.9,
        "markers_visible": True,
        "marker_color": "#ff3333",
        "marker_thickness": 2,
        "marker_alpha": 0.95,
        "stroke_scale_mode": "screen",
        "live_preview": False,
    }

    _LAYER_ROWS = (
        ("contour", "Contour", "contour", True),
        ("fit", "Strict fit", "fit", False),
        ("axes", "Axis", "axis", False),
        ("baseline", "Baseline", "baseline", False),
        ("markers", "Markers", "marker", False),
        ("points", "Points", "point", False),
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Overlay appearance")
        self._layer_controls: dict[str, dict[str, Any]] = {}
        self._build_ui()
        self._wire_signals()
        self.set_config(self.DEFAULTS)

    def _build_ui(self) -> None:
        self.setMinimumSize(680, 520)
        self.resize(760, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(14)
        top.addWidget(self._build_layer_table(), stretch=3)
        top.addWidget(self._build_preview_panel(), stretch=2)
        outer.addLayout(top, stretch=1)

        global_bar = QHBoxLayout()
        global_bar.setSpacing(10)
        global_bar.addWidget(QLabel("Stroke scaling:", self))
        self.stroke_scale_mode = QComboBox(self)
        self.stroke_scale_mode.addItem("Screen-constant", "screen")
        self.stroke_scale_mode.addItem("Image-scaled", "image")
        self.stroke_scale_mode.setMinimumWidth(160)
        global_bar.addWidget(self.stroke_scale_mode)
        global_bar.addSpacing(16)
        self.live_preview = QCheckBox("Live preview", self)
        global_bar.addWidget(self.live_preview)
        global_bar.addStretch(1)
        outer.addLayout(global_bar)

        button_bar = QHBoxLayout()
        self._preview_btn = QPushButton("Preview", self)
        self._preview_btn.setAutoDefault(False)
        self._reset_btn = QPushButton("Restore Defaults", self)
        self._reset_btn.setAutoDefault(False)
        button_bar.addWidget(self._preview_btn)
        button_bar.addWidget(self._reset_btn)
        button_bar.addStretch(1)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply,
            parent=self,
        )
        button_bar.addWidget(self._button_box)
        outer.addLayout(button_bar)

        ok_button = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setDefault(True)

    def _build_layer_table(self) -> QGroupBox:
        group = QGroupBox("Layers", self)
        outer = QVBoxLayout(group)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        grid_host = QWidget(group)
        grid_host.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout = QGridLayout(grid_host)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        headers = ("Layer", "Show", "Color", "Size", "Alpha", "Options")
        for column, text in enumerate(headers):
            label = QLabel(text, group)
            label.setStyleSheet("font-weight: 600;")
            layout.addWidget(label, 0, column)

        for row, (layer, title, prefix, supports_dash) in enumerate(
            self._LAYER_ROWS, start=1
        ):
            self._add_layer_row(layout, row, layer, title, prefix, supports_dash)

        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(5, 3)
        grid_host.setFixedHeight(grid_host.sizeHint().height())
        outer.addWidget(grid_host)
        outer.addStretch(1)
        return group

    def _add_layer_row(
        self,
        layout: QGridLayout,
        row: int,
        layer: str,
        label: str,
        prefix: str,
        supports_dash: bool = False,
    ) -> None:
        name = QLabel(label, self)
        visible = QCheckBox(self)

        color_btn = QPushButton(self)
        color_btn.setObjectName(f"{layer}ColorButton")
        color_btn.setFixedSize(42, 24)
        color_btn.setToolTip(f"{label} color")

        size = QSpinBox(self)
        size.setRange(1, 50 if layer == "points" else 20)
        size.setFixedWidth(72)

        alpha = QDoubleSpinBox(self)
        alpha.setRange(0.0, 1.0)
        alpha.setSingleStep(0.05)
        alpha.setDecimals(2)
        alpha.setFixedWidth(82)

        options = QWidget(self)
        options_layout = QHBoxLayout(options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(6)

        controls: dict[str, Any] = {
            "prefix": prefix,
            "visible": visible,
            "color_btn": color_btn,
            "color_preview": color_btn,
            "thickness": size,
            "alpha": alpha,
            "_color": QColor(str(self.DEFAULTS.get(f"{prefix}_color", "#ffffff"))),
        }

        if supports_dash:
            dashed = QCheckBox("Dash", self)
            dash_length_label = QLabel("Len", self)
            dash_length = QSpinBox(self)
            dash_length.setRange(1, 100)
            dash_length.setFixedWidth(58)
            dash_space_label = QLabel("Gap", self)
            dash_space = QSpinBox(self)
            dash_space.setRange(1, 100)
            dash_space.setFixedWidth(58)
            options_layout.addWidget(dashed)
            options_layout.addWidget(dash_length_label)
            options_layout.addWidget(dash_length)
            options_layout.addWidget(dash_space_label)
            options_layout.addWidget(dash_space)
            options_layout.addStretch(1)
            controls.update(
                {
                    "dashed": dashed,
                    "dash_length_label": dash_length_label,
                    "dash_length": dash_length,
                    "dash_space_label": dash_space_label,
                    "dash_space": dash_space,
                }
            )
            dashed.toggled.connect(self._update_dash_controls)
        else:
            options_layout.addWidget(QLabel("Default", self))
            options_layout.addStretch(1)

        color_btn.clicked.connect(
            lambda _checked=False, layer_name=layer: self._choose_layer_color(
                layer_name
            )
        )
        self._layer_controls[layer] = controls

        layout.addWidget(name, row, 0)
        layout.addWidget(visible, row, 1, alignment=Qt.AlignCenter)
        layout.addWidget(color_btn, row, 2)
        layout.addWidget(size, row, 3)
        layout.addWidget(alpha, row, 4)
        layout.addWidget(options, row, 5)

        if layer == "contour":
            self.contour_visible = visible
            self._contour_color_btn = color_btn
            self._contour_color_preview = color_btn
            self.contour_thickness = size
            self.contour_alpha = alpha
            self.contour_dashed = controls["dashed"]
            self.contour_dash_length = controls["dash_length"]
            self.contour_dash_space = controls["dash_space"]
        elif layer == "points":
            self.points_visible = visible
            self._point_color_btn = color_btn
            self._point_color_preview = color_btn
            self.point_radius = size
            self.point_alpha = alpha

    def _build_preview_panel(self) -> QGroupBox:
        group = QGroupBox("Example preview", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.preview_label = QLabel(self)
        self.preview_label.setObjectName("overlayExamplePreview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(250, 210)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_label.setStyleSheet("background: #eeeeee; color: #555;")
        layout.addWidget(self.preview_label)

        hint = QLabel(
            "Screen-constant strokes stay readable while zooming.", group
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)
        return group

    def _wire_signals(self) -> None:
        self._button_box.accepted.connect(self._on_ok)
        self._button_box.rejected.connect(self.reject)
        apply_btn = self._button_box.button(QDialogButtonBox.StandardButton.Apply)
        if apply_btn is not None:
            apply_btn.clicked.connect(self._on_apply)

        self._preview_btn.clicked.connect(self._on_preview)
        self._reset_btn.clicked.connect(self._on_reset)
        self.stroke_scale_mode.currentIndexChanged.connect(
            self._maybe_render_live_preview
        )
        self.live_preview.toggled.connect(self._maybe_render_live_preview)

        for controls in self._layer_controls.values():
            controls["visible"].toggled.connect(self._maybe_emit_live_preview)
            controls["thickness"].valueChanged.connect(self._maybe_emit_live_preview)
            controls["alpha"].valueChanged.connect(self._maybe_emit_live_preview)
            if "dashed" in controls:
                controls["dashed"].toggled.connect(self._maybe_emit_live_preview)
                controls["dash_length"].valueChanged.connect(self._maybe_emit_live_preview)
                controls["dash_space"].valueChanged.connect(self._maybe_emit_live_preview)

    def _on_ok(self) -> None:
        cfg = self.get_config()
        self.configApplied.emit(cfg)
        self.accept()

    def _on_apply(self) -> None:
        cfg = self.get_config()
        self.configApplied.emit(cfg)
        self._render_example_preview(cfg)

    def _on_preview(self) -> None:
        cfg = self.get_config()
        self._render_example_preview(cfg)
        self.previewRequested.emit(cfg)

    def _on_reset(self) -> None:
        self.set_config(self.DEFAULTS)

    def _maybe_emit_live_preview(self, *args, **kwargs) -> None:
        self._maybe_render_live_preview()
        if self.live_preview.isChecked():
            self.previewRequested.emit(self.get_config())

    def _maybe_render_live_preview(self, *args, **kwargs) -> None:
        self._render_example_preview(self.get_config())

    def get_config(self) -> Dict[str, Any]:
        cfg = self._collect_layer_config()
        cfg["live_preview"] = bool(self.live_preview.isChecked())
        cfg["stroke_scale_mode"] = self.stroke_scale_mode.currentData() or "screen"
        return cfg

    def set_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        merged = {**self.DEFAULTS, **(cfg or {})}
        self._apply_config_to_widgets(merged)
        self._render_example_preview(self.get_config())

    def _collect_layer_config(self) -> Dict[str, Any]:
        contour = self._layer_controls["contour"]
        points = self._layer_controls["points"]
        cfg: Dict[str, Any] = {
            "contour_visible": bool(contour["visible"].isChecked()),
            "contour_color": contour["_color"].name(),
            "contour_thickness": int(contour["thickness"].value()),
            "contour_dashed": bool(contour["dashed"].isChecked()),
            "contour_dash_length": int(contour["dash_length"].value()),
            "contour_dash_space": int(contour["dash_space"].value()),
            "contour_alpha": float(contour["alpha"].value()),
            "points_visible": bool(points["visible"].isChecked()),
            "point_color": points["_color"].name(),
            "point_radius": int(points["thickness"].value()),
            "point_alpha": float(points["alpha"].value()),
        }
        for layer in ("fit", "axes", "baseline", "markers"):
            controls = self._layer_controls[layer]
            prefix = controls["prefix"]
            cfg[f"{layer}_visible"] = bool(controls["visible"].isChecked())
            cfg[f"{prefix}_color"] = controls["_color"].name()
            cfg[f"{prefix}_thickness"] = int(controls["thickness"].value())
            cfg[f"{prefix}_alpha"] = float(controls["alpha"].value())
        return cfg

    def _apply_config_to_widgets(self, cfg: Dict[str, Any]) -> None:
        self._set_layer_values(
            "contour",
            visible=cfg["contour_visible"],
            color=QColor(str(cfg["contour_color"])),
            thickness=cfg["contour_thickness"],
            alpha=cfg["contour_alpha"],
        )
        self.contour_dashed.setChecked(bool(cfg["contour_dashed"]))
        self.contour_dash_length.setValue(int(cfg["contour_dash_length"]))
        self.contour_dash_space.setValue(int(cfg["contour_dash_space"]))

        self._set_layer_values(
            "points",
            visible=cfg["points_visible"],
            color=QColor(str(cfg["point_color"])),
            thickness=cfg["point_radius"],
            alpha=cfg["point_alpha"],
        )
        for layer in ("fit", "axes", "baseline", "markers"):
            controls = self._layer_controls[layer]
            prefix = controls["prefix"]
            self._set_layer_values(
                layer,
                visible=cfg.get(f"{layer}_visible", True),
                color=QColor(str(cfg.get(f"{prefix}_color", "#ffffff"))),
                thickness=cfg.get(f"{prefix}_thickness", 2),
                alpha=cfg.get(f"{prefix}_alpha", 1.0),
            )

        mode = str(cfg.get("stroke_scale_mode", self.DEFAULTS["stroke_scale_mode"]))
        self.stroke_scale_mode.setCurrentIndex(1 if mode == "image" else 0)
        self.live_preview.setChecked(bool(cfg.get("live_preview", False)))
        self._update_dash_controls()

    def _set_layer_values(
        self,
        layer: str,
        *,
        visible: Any,
        color: QColor,
        thickness: Any,
        alpha: Any,
    ) -> None:
        controls = self._layer_controls[layer]
        controls["visible"].setChecked(bool(visible))
        self._set_layer_color(layer, color if color.isValid() else QColor("white"))
        controls["thickness"].setValue(int(thickness))
        controls["alpha"].setValue(float(alpha))

    def _choose_layer_color(self, layer: str) -> None:
        controls = self._layer_controls[layer]
        color = QColorDialog.getColor(controls["_color"], self)
        if color.isValid():
            self._set_layer_color(layer, color)
            self._maybe_emit_live_preview()

    def _set_layer_color(self, layer: str, color: QColor) -> None:
        controls = self._layer_controls[layer]
        controls["_color"] = color
        controls["color_preview"].setStyleSheet(
            "QPushButton {"
            f"background-color: {color.name()};"
            "border: 1px solid #555;"
            "border-radius: 3px;"
            "}"
        )

    def _update_dash_controls(self) -> None:
        controls = self._layer_controls["contour"]
        enabled = bool(controls["dashed"].isChecked())
        for key in (
            "dash_length_label",
            "dash_length",
            "dash_space_label",
            "dash_space",
        ):
            controls[key].setVisible(enabled)
            controls[key].setEnabled(enabled)

    def _render_example_preview(self, config: Dict[str, Any]) -> None:
        pixmap = QPixmap(320, 230)
        pixmap.fill(QColor("#eeeeee"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        drop_path = QPainterPath()
        drop_path.moveTo(160, 45)
        drop_path.cubicTo(103, 70, 98, 160, 160, 190)
        drop_path.cubicTo(222, 160, 217, 70, 160, 45)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#242424")))
        painter.drawPath(drop_path)

        if config.get("baseline_visible", True):
            self._draw_preview_line(
                painter,
                (62, 190),
                (258, 190),
                QColor(str(config.get("baseline_color", "#ff00ff"))),
                float(config.get("baseline_thickness", 2)),
                float(config.get("baseline_alpha", 1)),
            )
        if config.get("axes_visible", True):
            self._draw_preview_line(
                painter,
                (160, 28),
                (160, 206),
                QColor(str(config.get("axis_color", "#00ffff"))),
                float(config.get("axis_thickness", 1)),
                float(config.get("axis_alpha", 1)),
            )
        if config.get("contour_visible", True):
            pen = self._preview_pen(
                QColor(str(config.get("contour_color", "#ff0000"))),
                float(config.get("contour_thickness", 2)),
                float(config.get("contour_alpha", 1)),
            )
            if config.get("contour_dashed"):
                pen.setDashPattern(
                    [
                        float(config.get("contour_dash_length", 6)),
                        float(config.get("contour_dash_space", 6)),
                    ]
                )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawPath(drop_path)
        if config.get("fit_visible", True):
            fit_path = QPainterPath()
            fit_path.moveTo(160, 48)
            fit_path.cubicTo(112, 78, 110, 155, 160, 184)
            fit_path.cubicTo(210, 155, 208, 78, 160, 48)
            painter.setPen(
                self._preview_pen(
                    QColor(str(config.get("fit_color", "#00aa00"))),
                    float(config.get("fit_thickness", 2)),
                    float(config.get("fit_alpha", 1)),
                )
            )
            painter.drawPath(fit_path)
        if config.get("markers_visible", True):
            marker_color = QColor(str(config.get("marker_color", "#ff3333")))
            marker_color.setAlphaF(
                max(0.0, min(1.0, float(config.get("marker_alpha", 1))))
            )
            painter.setPen(self._preview_pen(marker_color, float(config.get("marker_thickness", 2)), 1))
            painter.drawLine(151, 185, 169, 203)
            painter.drawLine(151, 203, 169, 185)
        if config.get("points_visible", True):
            point_color = QColor(str(config.get("point_color", "#00ff00")))
            point_color.setAlphaF(
                max(0.0, min(1.0, float(config.get("point_alpha", 1))))
            )
            radius = float(config.get("point_radius", 5))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(point_color))
            painter.drawEllipse(104 - radius, 105 - radius, radius * 2, radius * 2)
            painter.drawEllipse(216 - radius, 105 - radius, radius * 2, radius * 2)

        painter.end()
        self.preview_label.setPixmap(pixmap)

    def _draw_preview_line(
        self,
        painter: QPainter,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: QColor,
        width: float,
        alpha: float,
    ) -> None:
        painter.setPen(self._preview_pen(color, width, alpha))
        painter.drawLine(p1[0], p1[1], p2[0], p2[1])

    def _preview_pen(self, color: QColor, width: float, alpha: float) -> QPen:
        color = QColor(color)
        color.setAlphaF(max(0.0, min(1.0, float(alpha))))
        pen = QPen(color)
        pen.setWidthF(max(1.0, float(width)))
        pen.setCosmetic(True)
        return pen

    def _on_preview_image_ready(self, image: np.ndarray, metadata: dict) -> None:
        """Keep compatibility with old preview feeds; dialog uses generated preview."""
        self._render_example_preview(self.get_config())


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dlg = OverlayConfigDialog()
    dlg.show()
    sys.exit(app.exec())
