"""Geometry configuration dialog for Menipy.

This dialog exposes geometry/edge detector options discovered in plugins
like `circle_edge` and `sine_edge` and provides a small API to get/set the
configuration as a dict. It's intentionally lightweight and uses PySide6.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from PySide6.QtGui import QImage, QPixmap


class GeometryConfigDialog(QDialog):
    """Dialog to configure geometry/edge detector options.

    Methods
    -------
    get_config() -> dict: return the current config
    set_config(cfg: dict) -> None: apply a config dictionary to widgets

    Signal
    ------
    configApplied(dict) -- emitted when the user clicks OK or Apply
    """

    configApplied = Signal(dict)
    previewRequested = Signal(object)

    DEFAULTS = {
        "detector": "circle",
        "use_canny": False,
        "waves": 4,
        "amplitude": 0.07,
        "points": 300,
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Geometry configuration")
        self._build_ui()
        self._wire_signals()
        self.set_config(self.DEFAULTS)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()

        # detector choice (plugins register names like 'circle', 'sine')
        self.detectorCombo = QComboBox(self)
        # Keep common detectors; plugin discovery can overwrite these later
        self.detectorCombo.addItems(["circle", "sine"])
        form.addRow("Edge detector:", self.detectorCombo)

        # use Canny edge detector when available
        self.useCanny = QCheckBox(self)
        form.addRow("Use Canny:", self.useCanny)

        # waves (for sine-like detectors)
        self.wavesSpin = QSpinBox(self)
        self.wavesSpin.setRange(1, 100)
        form.addRow("Waves:", self.wavesSpin)

        # Option to preview using preprocessed image
        self.usePreprocessed = QCheckBox(self)
        form.addRow("Use preprocessed image for preview:", self.usePreprocessed)

        # amplitude (for sine modulation)
        self.amplitudeSpin = QDoubleSpinBox(self)
        self.amplitudeSpin.setDecimals(3)
        self.amplitudeSpin.setRange(0.0, 10.0)
        self.amplitudeSpin.setSingleStep(0.01)
        form.addRow("Amplitude:", self.amplitudeSpin)

        # points (number of contour points)
        self.pointsSpin = QSpinBox(self)
        self.pointsSpin.setRange(10, 5000)
        form.addRow("Points:", self.pointsSpin)

        layout.addLayout(form)

        # small info label for center/preview (read-only)
        self.infoLabel = QLabel("", self)
        self.infoLabel.setWordWrap(True)
        layout.addWidget(self.infoLabel)

        # Preview area (similar to preprocessing dialog)
        self.preview_label = QLabel("No preview available", self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 240)
        self.preview_label.setStyleSheet("background-color: #333; color: #CCC;")
        self.preview_label.setScaledContents(True)
        layout.addWidget(self.preview_label)

        # buttons + preview/reset
        button_bar = QHBoxLayout()
        self._preview_btn = QPushButton("Preview", self)
        self._preview_btn.setAutoDefault(False)
        self._reset_btn = QPushButton("Restore Defaults", self)
        self._reset_btn.setAutoDefault(False)
        button_bar.addWidget(self._preview_btn)
        button_bar.addWidget(self._reset_btn)
        button_bar.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply,
            parent=self,
        )
        button_bar.addWidget(buttons)
        hl = QHBoxLayout()
        layout.addLayout(button_bar)
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setDefault(True)
        # keep existing connections to accept/apply via wire_signals

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

    def _wire_signals(self) -> None:
        # Connect dialog button roles and preview/reset
        # Find the button box we created in _build_ui
        for child in self.findChildren(QDialogButtonBox):
            btn_box = child
            break
        else:
            btn_box = None

        if btn_box is not None:
            btn_box.accepted.connect(self._on_ok)
            btn_box.rejected.connect(self.reject)
            apply_btn = btn_box.button(QDialogButtonBox.StandardButton.Apply)
            if apply_btn is not None:
                apply_btn.clicked.connect(self._on_apply)

        try:
            self._preview_btn.clicked.connect(self._on_preview)
            self._reset_btn.clicked.connect(self._on_reset)
        except Exception:
            pass

    def get_config(self) -> Dict[str, Any]:
        return {
            "detector": str(self.detectorCombo.currentText()),
            "use_canny": bool(self.useCanny.isChecked()),
            "waves": int(self.wavesSpin.value()),
            "amplitude": float(self.amplitudeSpin.value()),
            "points": int(self.pointsSpin.value()),
            "use_preprocessed": bool(
                getattr(self, "usePreprocessed", None)
                and self.usePreprocessed.isChecked()
            ),
        }

    def set_config(self, cfg: Dict[str, Any]) -> None:
        if not cfg:
            return
        detector = cfg.get("detector")
        if detector:
            idx = self.detectorCombo.findText(str(detector))
            if idx == -1:
                self.detectorCombo.addItem(str(detector))
                idx = self.detectorCombo.count() - 1
            self.detectorCombo.setCurrentIndex(idx)
        if "use_canny" in cfg:
            self.useCanny.setChecked(bool(cfg.get("use_canny")))
        # Use provided values or fall back to sensible defaults to avoid None
        try:
            self.wavesSpin.setValue(int(cfg.get("waves", self.DEFAULTS["waves"])))
        except Exception:
            pass
        try:
            self.amplitudeSpin.setValue(
                float(cfg.get("amplitude", self.DEFAULTS["amplitude"]))
            )
        except Exception:
            pass
        try:
            self.pointsSpin.setValue(int(cfg.get("points", self.DEFAULTS["points"])))
        except Exception:
            pass
        if "use_preprocessed" in cfg:
            try:
                self.usePreprocessed.setChecked(bool(cfg.get("use_preprocessed")))
            except Exception:
                pass

    def set_info(self, text: str) -> None:
        self.infoLabel.setText(text)

    def _on_preview_image_ready(self, image: np.ndarray, metadata: dict) -> None:
        """Slot to receive preview images (QImage conversion similar to preprocessing dialog)."""
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


# Lightweight test harness when running module directly
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dlg = GeometryConfigDialog()
    dlg.show()
    sys.exit(app.exec())
