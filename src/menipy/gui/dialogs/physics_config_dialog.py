# src/menipy/gui/dialogs/physics_config_dialog.py
"""Dialog for editing physics parameters with unit support."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from menipy.models.config import PhysicsParams


class PhysicsConfigDialog(QDialog):
    """A dialog to configure PhysicsParams with unit-aware inputs."""

    def __init__(self, initial_params: PhysicsParams, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Physics Parameters")

        self.params = initial_params

        # Create widgets for each parameter
        self.delta_rho_edit = QLineEdit()
        self.surface_tension_guess_edit = QLineEdit()
        self.needle_radius_edit = QLineEdit()
        self.tube_radius_edit = QLineEdit()

        # Set initial text from the Pint quantities
        if self.params.delta_rho:
            self.delta_rho_edit.setText(str(self.params.delta_rho))
        if self.params.surface_tension_guess:
            self.surface_tension_guess_edit.setText(
                str(self.params.surface_tension_guess)
            )
        if self.params.needle_radius:
            self.needle_radius_edit.setText(str(self.params.needle_radius))
        if self.params.tube_radius:
            self.tube_radius_edit.setText(str(self.params.tube_radius))

        # Layout
        form_layout = QFormLayout()
        form_layout.addRow("Density Difference (Δρ):", self.delta_rho_edit)
        form_layout.addRow("Surface Tension Guess:", self.surface_tension_guess_edit)
        form_layout.addRow("Needle Radius:", self.needle_radius_edit)
        form_layout.addRow("Tube Radius (for Capillary Rise):", self.tube_radius_edit)

        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def accept(self):
        """Validate inputs and update the PhysicsParams model."""
        try:
            # Create a dictionary with the new values. Pydantic will parse them.
            new_data = {
                "delta_rho": self.delta_rho_edit.text() or None,
                "surface_tension_guess": self.surface_tension_guess_edit.text() or None,
                "needle_radius": self.needle_radius_edit.text() or None,
                "tube_radius": self.tube_radius_edit.text() or None,
                # Keep gravity and temperature as they were
                "g": self.params.g,
                "temperature_C": self.params.temperature_C,
            }
            # Let Pydantic do the parsing and validation
            self.params = PhysicsParams(**new_data)
            super().accept()
        except Exception as e:
            QMessageBox.critical(
                self, "Invalid Input", f"Error parsing parameters:\n{e}"
            )

    def get_params(self) -> PhysicsParams:
        """Returns the updated parameters after the dialog is accepted."""
        return self.params
