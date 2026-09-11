"""Pipeline-specific analysis settings widgets."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

PHASE_PROPERTIES_NOTE = (
    "Densities and gravity are set once, in Phase Properties on the setup "
    "panel (with the material database), and apply to every method."
)


def phase_properties_note() -> QLabel:
    """Hint pointing to the single place where densities and gravity are set.

    Returns
    -------
    QLabel
        Word-wrapped, muted label for the top of a settings form.
    """
    label = QLabel(PHASE_PROPERTIES_NOTE)
    label.setWordWrap(True)
    label.setStyleSheet("color: #7f8c8d;")
    return label
