"""Alternative contact angle analysis tab."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from .controls import AnalysisTab


class ContactAngleTabAlt(AnalysisTab):
    """Enhanced detection tab derived from :class:`AnalysisTab`."""

    def __init__(self, parent=None) -> None:
        super().__init__(show_contact_angle=True, parent=parent)
        self.debug_overlay_alt = False
        self.side_button = QPushButton("Select Drop Side")
        self.layout().insertRow(1, self.side_button)

