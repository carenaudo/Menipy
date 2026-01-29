"""
Pipeline-specific settings panel for the pendant drop pipeline.
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QTextEdit, QVBoxLayout, QComboBox, QCheckBox


class PipelineSettingsWidget(QWidget):
    """Physics + stage selections for pendant drop."""

    def __init__(self, parent=None, settings: dict | None = None):
        super().__init__(parent)
        settings = settings or {}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)

        self._rho1 = QDoubleSpinBox()
        self._rho1.setRange(0, 50000)
        self._rho1.setValue(1000.0)
        self._rho1.setSuffix(" kg/m³")
        if "rho1" in settings:
            self._rho1.setValue(float(settings["rho1"]))
        form.addRow("Liquid density", self._rho1)

        self._rho2 = QDoubleSpinBox()
        self._rho2.setRange(0, 50000)
        self._rho2.setValue(1.2)
        self._rho2.setSuffix(" kg/m³")
        if "rho2" in settings:
            self._rho2.setValue(float(settings["rho2"]))
        form.addRow("Ambient density", self._rho2)

        self._g = QDoubleSpinBox()
        self._g.setRange(0, 50)
        self._g.setDecimals(4)
        self._g.setValue(9.80665)
        self._g.setSuffix(" m/s²")
        if "g" in settings:
            self._g.setValue(float(settings["g"]))
        form.addRow("Gravity", self._g)

        # Solver selection
        self._solver = QComboBox()
        from menipy.common import registry

        solvers = sorted(registry.SOLVERS.keys()) or ["young_laplace_adsa"]
        self._solver.addItems(solvers)
        if "solver" in settings and settings["solver"] in solvers:
            self._solver.setCurrentText(settings["solver"])
        form.addRow("Solver", self._solver)

        # Preprocessor / edge detector choices
        self._preproc = QComboBox()
        preprocs = sorted(registry.PREPROCESSORS.keys()) or ["auto"]
        self._preproc.addItems(preprocs)
        if settings.get("preprocessor") in preprocs:
            self._preproc.setCurrentText(settings["preprocessor"])
        form.addRow("Preprocessor", self._preproc)

        self._edge = QComboBox()
        edges = sorted(registry.EDGE_DETECTORS.keys()) or ["canny"]
        self._edge.addItems(edges)
        if settings.get("edge_detector") in edges:
            self._edge.setCurrentText(settings["edge_detector"])
        form.addRow("Edge detector", self._edge)

        # Overlay options
        self._overlay_alpha = QDoubleSpinBox()
        self._overlay_alpha.setRange(0.0, 1.0)
        self._overlay_alpha.setSingleStep(0.05)
        self._overlay_alpha.setValue(float(settings.get("overlay_alpha", 0.6)))
        form.addRow("Overlay opacity", self._overlay_alpha)

        self._ov_master = QCheckBox("Show overlay")
        self._ov_master.setChecked(settings.get("overlay_visible", True))
        form.addRow("", self._ov_master)

        self._ov_contour = QCheckBox("Show contour")
        self._ov_contour.setChecked(settings.get("contour_visible", True))
        form.addRow("", self._ov_contour)

        self._ov_axis = QCheckBox("Show axis/needle")
        self._ov_axis.setChecked(settings.get("axis_visible", True))
        form.addRow("", self._ov_axis)

        self._ov_fit = QCheckBox("Show fitted profile/text")
        self._ov_fit.setChecked(settings.get("fit_visible", True))
        form.addRow("", self._ov_fit)

        self._notes = QTextEdit()
        self._notes.setPlaceholderText("Notes / preset name")
        self._notes.setFixedHeight(80)
        if settings.get("notes"):
            self._notes.setPlainText(settings["notes"])
        form.addRow("Notes", self._notes)

        layout.addLayout(form)
        layout.addStretch(1)

    def get_settings(self) -> dict:
        return {
            "rho1": float(self._rho1.value()),
            "rho2": float(self._rho2.value()),
            "g": float(self._g.value()),
            "solver": self._solver.currentText(),
            "notes": self._notes.toPlainText().strip() or None,
            "overlay_alpha": float(self._overlay_alpha.value()),
            "overlay_visible": self._ov_master.isChecked(),
            "contour_visible": self._ov_contour.isChecked(),
            "axis_visible": self._ov_axis.isChecked(),
            "fit_visible": self._ov_fit.isChecked(),
            "preprocessor": self._preproc.currentText(),
            "edge_detector": self._edge.currentText(),
        }
