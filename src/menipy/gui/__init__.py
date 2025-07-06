"""GUI package initialization."""

# Export the refactored UI as the default MainWindow
from ..ui import MainWindow
from .image_view import ImageView
from .calibration_dialog import CalibrationDialog
from .controls import (
    ZoomControl,
    ParameterPanel,
    MetricsPanel,
    CalibrationTab,
    AnalysisTab,
)
from .overlay import draw_drop_overlay
from .items import SubstrateLineItem


def main() -> None:
    """Launch the Menipy GUI application."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.showMaximized()
    app.exec()

__all__ = [
    "MainWindow",
    "ImageView",
    "CalibrationDialog",
    "ZoomControl",
    "ParameterPanel",
    "MetricsPanel",
    "CalibrationTab",
    "AnalysisTab",
    "draw_drop_overlay",
    "SubstrateLineItem",
    "main",
]
