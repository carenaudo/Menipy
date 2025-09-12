"""GUI package initialization."""

# Export the refactored UI as the default MainWindow
from menipy.ui import MainWindow
from menipy.zold_gui.image_view import ImageView
from menipy.zold_gui.calibration_dialog import CalibrationDialog
from menipy.zold_gui.controls import (
    ZoomControl,
    ParameterPanel,
    MetricsPanel,
    CalibrationTab,
    AnalysisTab,
)
from menipy.zold_gui.overlay import draw_drop_overlay
from menipy.zold_gui.items import SubstrateLineItem


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
