"""Dialog for overlay visualization settings."""

from __future__ import annotations

"""Dialog to configure overlay styling (contours, points, lines).

Provides a compact UI to tweak thickness, dashed stroke, dash pattern,
alpha/transparency, point radius, color preset, and visibility toggles.
The dialog exposes previewRequested(config) and configApplied(config) signals
and implements get_config()/set_config() so controllers can persist settings.
"""

from typing import Dict, Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QImage, QPixmap, QColor

import numpy as np


class OverlayConfigDialog(QDialog):
    """Configure overlay appearance.

    Signals
    -------
    previewRequested(dict) -- emitted when user clicks Preview
    configApplied(dict) -- emitted when user clicks OK or Apply
    """

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
    }

    COLOR_PRESETS = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "white",
        "black",
        "lime",
    ]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Overlay appearance")
        self._build_ui()
        self._wire_signals()
        self.set_config(self.DEFAULTS)

    def _build_ui(self) -> None:
        self.setMinimumSize(520, 420)
        outer = QVBoxLayout(self)
        form = QFormLayout()

        # Contour controls
        self.contour_visible = QCheckBox(self)
        form.addRow("Show contour:", self.contour_visible)

        self._contour_color_btn = QPushButton("Choose...", self)
        self._contour_color_preview = QLabel(self)
        self._contour_color_preview.setFixedSize(34, 18)
        form.addRow(
            "Contour color:",
            self._color_row(self._contour_color_btn, self._contour_color_preview),
        )
        self.contour_thickness = QSpinBox(self)
        self.contour_thickness.setRange(1, 20)
        form.addRow("Contour thickness (px):", self.contour_thickness)

        self.contour_dashed = QCheckBox(self)
        form.addRow("Use dashed stroke:", self.contour_dashed)

        self.contour_dash_length = QSpinBox(self)
        self.contour_dash_length.setRange(1, 100)
        form.addRow("Dash length (px):", self.contour_dash_length)

        self.contour_dash_space = QSpinBox(self)
        self.contour_dash_space.setRange(1, 100)
        form.addRow("Dash spacing (px):", self.contour_dash_space)

        self.contour_alpha = QDoubleSpinBox(self)
        self.contour_alpha.setRange(0.0, 1.0)
        self.contour_alpha.setSingleStep(0.05)
        form.addRow("Contour alpha:", self.contour_alpha)

        # Live preview toggle (not persisted by default)
        self.live_preview = QCheckBox(self)
        form.addRow("Live preview:", self.live_preview)

        # Points controls
        self.points_visible = QCheckBox(self)
        form.addRow("Show detected points:", self.points_visible)

        self._point_color_btn = QPushButton("Choose...", self)
        self._point_color_preview = QLabel(self)
        self._point_color_preview.setFixedSize(34, 18)
        form.addRow(
            "Point color:",
            self._color_row(self._point_color_btn, self._point_color_preview),
        )

        self.point_radius = QSpinBox(self)
        self.point_radius.setRange(1, 50)
        form.addRow("Point radius (px):", self.point_radius)

        self.point_alpha = QDoubleSpinBox(self)
        self.point_alpha.setRange(0.0, 1.0)
        self.point_alpha.setSingleStep(0.05)
        form.addRow("Point alpha:", self.point_alpha)

        outer.addLayout(form)

        # Preview area
        self.preview_label = QLabel("No preview available", self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 200)
        self.preview_label.setStyleSheet("background-color: #222; color: #DDD;")
        self.preview_label.setScaledContents(True)
        outer.addWidget(self.preview_label)

        # Buttons
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

    def _wire_signals(self) -> None:
        # Button box
        if self._button_box is not None:
            self._button_box.accepted.connect(self._on_ok)
            self._button_box.rejected.connect(self.reject)
            apply_btn = self._button_box.button(QDialogButtonBox.StandardButton.Apply)
            if apply_btn is not None:
                apply_btn.clicked.connect(self._on_apply)

        try:
            self._preview_btn.clicked.connect(self._on_preview)
            self._reset_btn.clicked.connect(self._on_reset)
            # Live preview: watch a subset of controls and re-emit previewRequested when changed
            self.live_preview.toggled.connect(lambda on: None)
            for w in (
                self.contour_thickness,
                self.contour_dashed,
                self.contour_dash_length,
                self.contour_dash_space,
                self.contour_alpha,
                self.point_radius,
                self.point_alpha,
                self.contour_visible,
                self.points_visible,
            ):
                try:
                    w.valueChanged.connect(self._maybe_emit_live_preview)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        w.toggled.connect(self._maybe_emit_live_preview)  # type: ignore[attr-defined]
                    except Exception:
                        pass
        except Exception:
            pass

    def _on_ok(self) -> None:
        cfg = self.get_config()
        self.configApplied.emit(cfg)
        self.accept()

    def _on_apply(self) -> None:
        cfg = self.get_config()
        self.configApplied.emit(cfg)

    def _on_preview(self) -> None:
        cfg = self.get_config()
        self.previewRequested.emit(cfg)

    def _on_reset(self) -> None:
        self.set_config(self.DEFAULTS)

    def _maybe_emit_live_preview(self, *args, **kwargs) -> None:
        try:
            if getattr(self, "live_preview", None) and self.live_preview.isChecked():
                cfg = self.get_config()
                # Emit previewRequested in the same way as pressing Preview
                try:
                    self.previewRequested.emit(cfg)
                except Exception:
                    pass
        except Exception:
            pass

    def get_config(self) -> Dict[str, Any]:
        """Get_config."""
        return {
            "contour_visible": bool(self.contour_visible.isChecked()),
            "contour_color": getattr(self, "_contour_color", QColor("red")).name(),
            "contour_thickness": int(self.contour_thickness.value()),
            "contour_dashed": bool(self.contour_dashed.isChecked()),
            "contour_dash_length": int(self.contour_dash_length.value()),
            "contour_dash_space": int(self.contour_dash_space.value()),
            "contour_alpha": float(self.contour_alpha.value()),
            "points_visible": bool(self.points_visible.isChecked()),
            "point_color": getattr(self, "_point_color", QColor("lime")).name(),
            "point_radius": int(self.point_radius.value()),
            "point_alpha": float(self.point_alpha.value()),
            "live_preview": bool(self.live_preview.isChecked()),
        }

    def set_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        """Set config.

        Parameters
        ----------
        cfg : type
        Description.
        """
        if not cfg:
            return
        try:
            self.contour_visible.setChecked(
                bool(cfg.get("contour_visible", self.DEFAULTS["contour_visible"]))
            )
            try:
                color = QColor(
                    str(cfg.get("contour_color", self.DEFAULTS["contour_color"]))
                )
            except Exception:
                color = QColor("red")
            self._set_contour_color(color)
            self.contour_thickness.setValue(
                int(cfg.get("contour_thickness", self.DEFAULTS["contour_thickness"]))
            )
            self.contour_dashed.setChecked(
                bool(cfg.get("contour_dashed", self.DEFAULTS["contour_dashed"]))
            )
            self.contour_dash_length.setValue(
                int(
                    cfg.get("contour_dash_length", self.DEFAULTS["contour_dash_length"])
                )
            )
            self.contour_dash_space.setValue(
                int(cfg.get("contour_dash_space", self.DEFAULTS["contour_dash_space"]))
            )
            self.contour_alpha.setValue(
                float(cfg.get("contour_alpha", self.DEFAULTS["contour_alpha"]))
            )

            self.points_visible.setChecked(
                bool(cfg.get("points_visible", self.DEFAULTS["points_visible"]))
            )
            try:
                pcol = QColor(str(cfg.get("point_color", self.DEFAULTS["point_color"])))
            except Exception:
                pcol = QColor("lime")
            self._set_point_color(pcol)
            self.point_radius.setValue(
                int(cfg.get("point_radius", self.DEFAULTS["point_radius"]))
            )
            self.point_alpha.setValue(
                float(cfg.get("point_alpha", self.DEFAULTS["point_alpha"]))
            )
        except Exception:
            # Defensive: ignore invalid values coming from older configs
            pass

    def _on_preview_image_ready(self, image: np.ndarray, metadata: dict) -> None:
        """Receive preview images (same approach as other dialogs).

        This is a light conversion for quick visual previews inside the dialog.
        """
        if image is None:
            self.preview_label.setText("No preview available")
            self.preview_label.clear()
            return

        h, w = image.shape[:2]
        bytes_per_line = 3 * w

        if image.ndim == 2:  # Grayscale
            q_image = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        elif image.ndim == 3 and image.shape[2] == 3:  # BGR
            q_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888
            )
        elif image.ndim == 3 and image.shape[2] == 4:  # BGRA
            q_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format.Format_ARGB32
            )
        else:
            self.preview_label.setText("Unsupported image format")
            self.preview_label.clear()
            return

        pixmap = QPixmap.fromImage(q_image)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # Helper UI for color selection row
    def _color_row(self, btn: QPushButton, preview: QLabel) -> QWidget:
        w = QWidget(self)
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        h.addWidget(btn)
        h.addWidget(preview)
        btn.clicked.connect(lambda: self._on_choose_color(btn))
        return w

    def _on_choose_color(self, source_btn: QPushButton) -> None:
        # Open QColorDialog and set appropriate preview
        dlg = QColorDialog(self)
        if source_btn is self._contour_color_btn:
            initial = getattr(self, "_contour_color", QColor("red"))
        else:
            initial = getattr(self, "_point_color", QColor("lime"))
        col = dlg.getColor(initial, parent=self)
        if not col.isValid():
            return
        if source_btn is self._contour_color_btn:
            self._set_contour_color(col)
        else:
            self._set_point_color(col)

    def _set_contour_color(self, qcolor: QColor) -> None:
        self._contour_color = qcolor
        self._contour_color_preview.setStyleSheet(
            f"background-color: {qcolor.name()}; border: 1px solid #222;"
        )

    def _set_point_color(self, qcolor: QColor) -> None:
        self._point_color = qcolor
        self._point_color_preview.setStyleSheet(
            f"background-color: {qcolor.name()}; border: 1px solid #222;"
        )


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dlg = OverlayConfigDialog()
    dlg.show()
    sys.exit(app.exec())
