"""
Pipeline-specific settings panel for the sessile drop pipeline.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QFormLayout,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QComboBox,
    QCheckBox,
)


class PipelineSettingsWidget(QWidget):
    """Geometry/physics selections for sessile drop."""

    def __init__(self, parent=None, settings: dict | None = None):
        super().__init__(parent)
        settings = settings or {}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)

        self._contact_method = QComboBox()
        self._contact_method.addItems(["tangent", "circle_fit", "spherical_cap"])
        if settings.get("contact_angle_method") in ["tangent", "circle_fit", "spherical_cap"]:
            self._contact_method.setCurrentText(settings["contact_angle_method"])
        form.addRow("Contact angle method", self._contact_method)

        self._rho1 = QDoubleSpinBox()
        self._rho1.setRange(0, 50000)
        self._rho1.setValue(float(settings.get("rho1", 1000.0)))
        self._rho1.setSuffix(" kg/m³")
        form.addRow("Liquid density", self._rho1)

        self._rho2 = QDoubleSpinBox()
        self._rho2.setRange(0, 50000)
        self._rho2.setValue(float(settings.get("rho2", 1.2)))
        self._rho2.setSuffix(" kg/m³")
        form.addRow("Ambient density", self._rho2)

        self._g = QDoubleSpinBox()
        self._g.setRange(0, 50)
        self._g.setDecimals(4)
        self._g.setValue(float(settings.get("g", 9.80665)))
        self._g.setSuffix(" m/s²")
        form.addRow("Gravity", self._g)

        self._solver = QComboBox()
        from menipy.common import registry

        solvers = sorted(registry.SOLVERS.keys()) or ["toy_young_laplace"]
        self._solver.addItems(solvers)
        if settings.get("solver") in solvers:
            self._solver.setCurrentText(settings["solver"])
        form.addRow("Solver", self._solver)

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

        self._ov_baseline = QCheckBox("Show baseline")
        self._ov_baseline.setChecked(settings.get("baseline_visible", True))
        form.addRow("", self._ov_baseline)

        self._ov_contact = QCheckBox("Show contact points")
        self._ov_contact.setChecked(settings.get("contact_visible", True))
        form.addRow("", self._ov_contact)

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
            "contact_angle_method": self._contact_method.currentText(),
            "rho1": float(self._rho1.value()),
            "rho2": float(self._rho2.value()),
            "g": float(self._g.value()),
            "solver": self._solver.currentText(),
            "notes": self._notes.toPlainText().strip() or None,
            "overlay_alpha": float(self._overlay_alpha.value()),
            "overlay_visible": self._ov_master.isChecked(),
            "baseline_visible": self._ov_baseline.isChecked(),
            "contact_visible": self._ov_contact.isChecked(),
            "preprocessor": self._preproc.currentText(),
            "edge_detector": self._edge.currentText(),
        }
