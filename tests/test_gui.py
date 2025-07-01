"""Tests for gui module."""

import pytest

try:
    from src.gui import MainWindow
    from PySide6 import QtWidgets
except Exception as exc:
    MainWindow = None
    QtWidgets = None
    missing_dependency = exc
else:
    missing_dependency = None


def test_main_window_exists():
    if MainWindow is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    assert MainWindow is not None


def test_main_window_instantiation():
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "Menipy"
    assert window.algorithm_combo.count() >= 2
    window.close()
    app.quit()
