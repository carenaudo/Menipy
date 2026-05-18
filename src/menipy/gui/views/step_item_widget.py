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
        self._included = True

        self.statusLbl = QLabel()
        self.statusLbl.setFixedSize(10, 10)
        self.statusLbl.setObjectName("statusDot")

        display_name = step_name.replace("_", " ").title()
        self.nameLbl = QLabel(display_name)
        self.nameLbl.setToolTip(step_name)
        self.nameLbl.setObjectName("stepNameLabel")
        self.nameLbl.setMinimumWidth(0)
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
        self.setProperty("included", True)
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

    def is_included(self) -> bool:
        return self._included

    def set_included(self, included: bool) -> None:
        self._included = bool(included)
        self.setProperty("included", self._included)
        self.playBtn.setEnabled(self._included)
        self.cfgBtn.setEnabled(self._included)
        if self._included:
            self.nameLbl.setToolTip(self._name)
            self._apply_status_style(self.property("stepState") or "pending")
        else:
            self.nameLbl.setToolTip(f"{self._name} excluded from current SOP")
            self.statusLbl.setStyleSheet(
                "background-color: #CBD5E1; border-radius: 5px;"
            )
            self.statusLbl.setToolTip("Excluded")
        self.style().unpolish(self)
        self.style().polish(self)

    def _apply_status_style(self, status: str) -> None:
        status = status if status in _STATUS_COLORS else "pending"
        self.setProperty("stepState", status)
        if not self._included:
            self.statusLbl.setStyleSheet(
                "background-color: #CBD5E1; border-radius: 5px;"
            )
            self.statusLbl.setToolTip("Excluded")
            self.style().unpolish(self)
            self.style().polish(self)
            return
        color = _STATUS_COLORS[status]
        self.statusLbl.setStyleSheet(
            f"background-color: {color}; border-radius: 5px;"
        )
        self.statusLbl.setToolTip(status.title())
        self.style().unpolish(self)
        self.style().polish(self)

    def _apply_widget_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget[stepState="pending"] {
                background: #ffffff;
                border: 1px solid #d7dde5;
                border-radius: 6px;
            }
            QWidget[stepState="running"] {
                background: #eef8ff;
                border: 1px solid #38bdf8;
                border-radius: 6px;
            }
            QWidget[stepState="done"] {
                background: #ecfdf5;
                border: 1px solid #34d399;
                border-radius: 6px;
            }
            QWidget[stepState="error"] {
                background: #fef2f2;
                border: 1px solid #f87171;
                border-radius: 6px;
            }
            QWidget[included="false"] {
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                border-radius: 6px;
            }
            QLabel#stepNameLabel {
                color: #1f2937;
                font-weight: 600;
            }
            QWidget[included="false"] QLabel#stepNameLabel {
                color: #64748b;
                font-weight: 500;
            }
            QToolButton {
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 2px;
            }
            QToolButton:hover {
                background: #e2e8f0;
            }
            QToolButton:pressed {
                background: #cbd5e1;
            }
            QToolButton:disabled {
                background: #f1f5f9;
                border-color: #e2e8f0;
            }
            """
        )
