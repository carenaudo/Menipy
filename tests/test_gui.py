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


def test_load_image_retains_size(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2

    img = np.zeros((20, 30, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    pixmap = window.pixmap_item.pixmap()
    assert pixmap.width() == 30
    assert pixmap.height() == 20
    assert window.graphics_view.width() == 30
    assert window.graphics_view.height() == 20

    transform = window.graphics_view.transform()
    assert transform.m11() == 1
    assert transform.m22() == 1

    window.close()
    app.quit()


def test_save_annotated_image(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2

    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:8, 2:8] = 255
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    window.process_image()

    out_path = tmp_path / "annotated.png"
    window.save_annotated_image(out_path)

    assert out_path.exists()

    window.close()
    app.quit()


def test_ml_segmentation_toggle(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2

    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:8, 2:8] = 255
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.use_ml_action.setChecked(True)
    window.load_image(path)
    window.process_image()

    assert window.mask_item is not None

    window.close()
    app.quit()


def test_zoom_control(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    window.zoom_control.slider.setValue(200)
    transform = window.graphics_view.transform()
    assert transform.m11() == pytest.approx(2.0, rel=0.01)
    assert transform.m22() == pytest.approx(2.0, rel=0.01)

    window.close()
    app.quit()
