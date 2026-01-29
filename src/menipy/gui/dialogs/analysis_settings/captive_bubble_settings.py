"""
Pipeline-specific settings panel for captive bubble pipeline (basic stub).
"""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QFormLayout, QDoubleSpinBox, QVBoxLayout, QComboBox, QTextEdit


class PipelineSettingsWidget(QWidget):
    """Minimal physics/solver selector for captive bubble."""

    def __init__(self, parent=None, settings: dict | None = None):
        super().__init__(parent)
        settings = settings or {}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)

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
        }
