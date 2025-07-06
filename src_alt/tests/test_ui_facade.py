import pytest

try:
    from menipy.ui import MainWindow
    from PySide6 import QtWidgets
except Exception as exc:  # PySide6 might be missing
    MainWindow = None
    QtWidgets = None
    missing_dependency = exc
else:
    missing_dependency = None


def test_main_window_facade_instantiates() -> None:
    if QtWidgets is None:
        pytest.skip(f"PySide6 not available: {missing_dependency}")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    assert hasattr(win, "detect_needle")
    win.close()
    app.quit()
