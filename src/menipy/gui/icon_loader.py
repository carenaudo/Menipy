"""Shared Qt icon loading helpers for GUI resources."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QAbstractButton

_ICON_DIR = Path(__file__).resolve().parent / "resources" / "icons"


def load_icon(name: str) -> QIcon:
    """Load an icon from Qt resources, falling back to the source tree."""
    icon_name = name if name.endswith(".svg") else f"{name}.svg"
    icon = QIcon(f":/icons/{icon_name}")
    if not icon.isNull():
        return icon

    fallback = _ICON_DIR / icon_name
    if fallback.exists():
        icon = QIcon(str(fallback))
        if not icon.isNull():
            return icon
    return QIcon()


def set_button_icon(
    button: QAbstractButton | None,
    name: str,
    *,
    size: int = 16,
    clear_text: bool = False,
) -> None:
    """Apply a named icon to a button when the icon can be loaded."""
    if button is None:
        return
    icon = load_icon(name)
    if icon.isNull():
        return
    button.setIcon(icon)
    button.setIconSize(QSize(size, size))
    if clear_text:
        button.setText("")
