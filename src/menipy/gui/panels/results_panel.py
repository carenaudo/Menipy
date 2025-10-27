"""Results panel helper for Menipy GUI."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtWidgets import QWidget, QTableWidget, QTableWidgetItem

LABEL_MAP = {
    "diameter_mm": "Diameter (mm)",
    "height_mm": "Height (mm)",
    "volume_uL": "Volume (µL)",
    "surface_tension_mN_m": "Surface Tension (mN/m)",
    "beta": "β (Shape Factor)",
    "s1": "s₁ (Shape Factor)",
    "r0_mm": "Apex Radius (mm)",
    "needle_surface_mm2": "Needle Surface (mm²)",
    "drop_surface_mm2": "Drop Surface (mm²)",
    "contact_angle_deg": "Contact Angle (°)",
    "contact_surface_mm2": "Contact Surface (mm²)",
    "theta_left_deg": "Left Contact Angle (°)",
    "theta_right_deg": "Right Contact Angle (°)",
    "baseline_tilt_deg": "Baseline Tilt (°)",
    "method": "Method",
    "uncertainty_deg": "Angle Uncertainty (°)",
    "uncertainty_left_deg": "Left Angle Uncertainty (°)",
    "uncertainty_right_deg": "Right Angle Uncertainty (°)",
}


class ResultsPanel:
    """Provides a helper wrapper around the results table widget."""

    def __init__(self, panel: QWidget) -> None:
        self.panel = panel
        self.table: Optional[QTableWidget] = panel.findChild(QTableWidget, "resultsTable")
        if self.table:
            self.table.setColumnCount(2)
            self.table.setHorizontalHeaderLabels(["Parameter", "Value"])

    def update(self, results: Mapping[str, Any] | None) -> None:
        if not self.table:
            return
        if not results:
            self.table.setRowCount(0)
            return

        # Special handling for uncertainty_deg which is a dict
        items = []
        for key, value in results.items():
            if key == "uncertainty_deg" and isinstance(value, dict):
                # Display left/right uncertainties separately
                if "left" in value:
                    items.append(("uncertainty_left_deg", value["left"]))
                if "right" in value:
                    items.append(("uncertainty_right_deg", value["right"]))
            else:
                items.append((key, value))

        self.table.setRowCount(len(items))
        for row, (key, value) in enumerate(items):
            label = LABEL_MAP.get(key, str(key).replace("_", " ").title())
            if isinstance(value, float):
                if "angle" in key.lower() or "tilt" in key.lower():
                    value_str = f"{value:.1f}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            self.table.setItem(row, 0, QTableWidgetItem(label))
            self.table.setItem(row, 1, QTableWidgetItem(value_str))
        self.table.resizeColumnsToContents()
