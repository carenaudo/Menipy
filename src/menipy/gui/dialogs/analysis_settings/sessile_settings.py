"""Pipeline-specific settings panel for the sessile drop pipeline."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from menipy.gui.dialogs.analysis_settings import phase_properties_note

CONTACT_ANGLE_METHODS = [
    "tangent",
    "circle_fit",
    "spherical_cap",
    "auto_residual",
    "arc_spline",
    "clothoid_spline",
]


class PipelineSettingsWidget(QWidget):
    """Method and stage selections for sessile drop."""

    def __init__(self, parent=None, settings: dict | None = None):
        super().__init__(parent)
        settings = settings or {}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)
        form.addRow(phase_properties_note())

        self._contact_method = QComboBox()
        self._contact_method.addItems(CONTACT_ANGLE_METHODS)
        if settings.get("contact_angle_method") in CONTACT_ANGLE_METHODS:
            self._contact_method.setCurrentText(settings["contact_angle_method"])
        form.addRow("Contact angle method", self._contact_method)

        self._experimental_mode = QComboBox()
        self._experimental_mode.addItems(["off", "shadow"])
        self._experimental_mode.setCurrentText(settings.get("experimental_geometry_mode", "off"))
        form.addRow("Experimental geometry", self._experimental_mode)

        self._needle_geometry = QComboBox()
        self._needle_geometry.addItems(["legacy", "bilateral_robust"])
        self._needle_geometry.setCurrentText(settings.get("needle_geometry_method", "legacy"))
        form.addRow("Needle geometry", self._needle_geometry)

        self._onnx_mode = QComboBox()
        self._onnx_mode.addItems(["off", "shadow"])
        self._onnx_mode.setCurrentText(settings.get("onnx_proposal_mode", "off"))
        form.addRow("ONNX proposals", self._onnx_mode)

        self._segmentation_provider = QComboBox()
        self._segmentation_provider.addItems(["mobilesam"])
        self._segmentation_provider.setCurrentText(
            settings.get("segmentation_provider", "mobilesam")
        )
        form.addRow("Segmentation provider", self._segmentation_provider)

        layout.addLayout(form)
        layout.addStretch(1)

    def get_settings(self) -> dict:
        """Collect the sessile pipeline settings from the widgets.

        Returns
        -------
        dict
            Flat pipeline settings; every key is forwarded to the run.
        """
        return {
            "contact_angle_method": self._contact_method.currentText(),
            "experimental_geometry_mode": self._experimental_mode.currentText(),
            "needle_geometry_method": self._needle_geometry.currentText(),
            "onnx_proposal_mode": self._onnx_mode.currentText(),
            "segmentation_provider": self._segmentation_provider.currentText(),
        }
