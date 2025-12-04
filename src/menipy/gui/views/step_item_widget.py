"""
Widget for displaying pipeline step items.
"""
from __future__ import annotations
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from pathlib import Path
from PySide6.QtWidgets import QWidget, QLabel, QToolButton, QHBoxLayout, QSizePolicy

def _load_icon(name: str) -> QIcon:
    """Try resource path first (:/icons/...), then fall back to on-disk resources/icons/*"""
    res_path = f":/icons/{name}"
    ico = QIcon(res_path)
    if not ico.isNull():
        return ico
    # fallback to repo file
    base = Path(__file__).resolve().parent.parent / "resources" / "icons"
    p = base / name
    if p.exists():
        ico2 = QIcon(str(p))
        if not ico2.isNull():
            return ico2
    return QIcon()

STATUS_ICONS = {
    "pending":  _load_icon("status_pending.svg"),
    "running":  _load_icon("status_running.svg"),
    "done":     _load_icon("status_done.svg"),
    "error":    _load_icon("status_error.svg"),
}

PLAY_ICON   = _load_icon("play.svg")
CONFIG_ICON = _load_icon("settings.svg")

class StepItemWidget(QWidget):
    playClicked = Signal(str)
    configClicked = Signal(str)

    def __init__(self, step_name: str, parent: Optional[QWidget]=None):
        super().__init__(parent)
        self._name = step_name

        self.statusLbl = QLabel()
        self.statusLbl.setPixmap(STATUS_ICONS.get("pending").pixmap(16, 16))

        self.nameLbl = QLabel(step_name)
        self.nameLbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.playBtn = QToolButton()
        self.playBtn.setIcon(PLAY_ICON)
        self.playBtn.setToolTip("Run this step")

        self.cfgBtn = QToolButton()
        self.cfgBtn.setIcon(CONFIG_ICON)
        self.cfgBtn.setToolTip("Configure this step")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(6)
        lay.addWidget(self.statusLbl)
        lay.addWidget(self.nameLbl, 1)
        lay.addWidget(self.playBtn)
        lay.addWidget(self.cfgBtn)

        self.playBtn.clicked.connect(lambda: self.playClicked.emit(self._name))
        self.cfgBtn.clicked.connect(lambda: self.configClicked.emit(self._name))

    @property
    def step_name(self) -> str:
        return self._name

    def set_status(self, status: str) -> None:
        icon = STATUS_ICONS.get(status, STATUS_ICONS["pending"])
        self.statusLbl.setPixmap(icon.pixmap(16, 16))
