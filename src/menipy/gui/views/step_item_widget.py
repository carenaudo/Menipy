"""Widget for displaying pipeline step items."""

# type: ignore
from __future__ import annotations
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtCore import QSize
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


PLAY_ICON = _load_icon("play.svg")
CONFIG_ICON = _load_icon("settings.svg")
_STATUS_COLORS = {
    "pending": "#64748B",
    "running": "#38BDF8",
    "done": "#34D399",
    "error": "#F87171",
}


class StepItemWidget(QWidget):
    playClicked = Signal(str)
    configClicked = Signal(str)

    def __init__(self, step_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._name = step_name

        self.statusLbl = QLabel()
        self.statusLbl.setFixedSize(10, 10)
        self.statusLbl.setObjectName("statusDot")

        display_name = step_name.replace("_", " ").title()
        self.nameLbl = QLabel(display_name)
        self.nameLbl.setToolTip(step_name)
        self.nameLbl.setObjectName("stepNameLabel")
        self.nameLbl.setMinimumWidth(100)
        self.nameLbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.playBtn = QToolButton()
        self.playBtn.setIcon(PLAY_ICON)
        if PLAY_ICON.isNull():
            self.playBtn.setText("Run")
        self.playBtn.setToolTip("Run this step")
        self.playBtn.setAutoRaise(True)
        self.playBtn.setIconSize(QSize(14, 14))
        self.playBtn.setFixedSize(24, 24)
        self.playBtn.setCursor(Qt.PointingHandCursor)

        self.cfgBtn = QToolButton()
        self.cfgBtn.setIcon(CONFIG_ICON)
        if CONFIG_ICON.isNull():
            self.cfgBtn.setText("Edit")
        self.cfgBtn.setToolTip("Configure this step")
        self.cfgBtn.setAutoRaise(True)
        self.cfgBtn.setIconSize(QSize(14, 14))
        self.cfgBtn.setFixedSize(24, 24)
        self.cfgBtn.setCursor(Qt.PointingHandCursor)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)
        lay.addWidget(self.statusLbl)
        lay.addWidget(self.nameLbl, 1)
        lay.addWidget(self.playBtn)
        lay.addWidget(self.cfgBtn)

        self.playBtn.clicked.connect(lambda: self.playClicked.emit(self._name))
        self.cfgBtn.clicked.connect(lambda: self.configClicked.emit(self._name))
        self.setMinimumHeight(34)
        self._apply_status_style("pending")
        self._apply_widget_style()

    @property
    def step_name(self) -> str:
        return self._name

    def set_status(self, status: str) -> None:
        """Set status.

        Parameters
        ----------
        status : type
        Description.
        """
        self._apply_status_style(status)

    def _apply_status_style(self, status: str) -> None:
        status = status if status in _STATUS_COLORS else "pending"
        color = _STATUS_COLORS[status]
        self.statusLbl.setStyleSheet(
            f"background-color: {color}; border-radius: 5px;"
        )
        self.statusLbl.setToolTip(status.title())
        self.setProperty("stepState", status)
        self.style().unpolish(self)
        self.style().polish(self)

    def _apply_widget_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget[stepState="pending"] {
                background: #1f1f1f;
                border: 1px solid #2a2a2a;
                border-radius: 6px;
            }
            QWidget[stepState="running"] {
                background: #1e2a36;
                border: 1px solid #2f4b66;
                border-radius: 6px;
            }
            QWidget[stepState="done"] {
                background: #1f2b24;
                border: 1px solid #2d6a47;
                border-radius: 6px;
            }
            QWidget[stepState="error"] {
                background: #2b1f1f;
                border: 1px solid #6a2d2d;
                border-radius: 6px;
            }
            QLabel#stepNameLabel {
                color: #e5e7eb;
                font-weight: 600;
            }
            QToolButton {
                background: #2b2b2b;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 2px;
            }
            QToolButton:hover {
                background: #3a3a3a;
            }
            QToolButton:pressed {
                background: #4a4a4a;
            }
            QToolButton:disabled {
                background: #1b1b1b;
                border-color: #2a2a2a;
            }
            """
        )
