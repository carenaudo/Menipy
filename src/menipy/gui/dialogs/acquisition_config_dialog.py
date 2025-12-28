"""
Dialog for configuring image acquisition settings.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)


class AcquisitionConfigDialog(QDialog):
    """Configuration dialog for the acquisition pipeline stage."""

    def __init__(self, *, contact_line_required: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Acquisition Configuration")
        layout = QVBoxLayout(self)

        layout.addWidget(
            QLabel("ROI definition is required for the acquisition stage.")
        )
        layout.addWidget(
            QLabel("Needle region definition is required for the acquisition stage.")
        )

        self._contact_line_checkbox = QCheckBox("Require contact line")
        self._contact_line_checkbox.setChecked(contact_line_required)
        layout.addWidget(self._contact_line_checkbox)
        layout.addWidget(
            QLabel("Select to require the contact line before running acquisition.")
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def contact_line_required(self) -> bool:
        return self._contact_line_checkbox.isChecked()
