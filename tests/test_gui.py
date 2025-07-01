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


def test_parameter_panel_defaults():
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    panel = window.parameter_panel
    values = panel.values()

    assert "air_density" in values and values["air_density"] > 0
    assert "liquid_density" in values and values["liquid_density"] > 0
    assert "surface_tension" in values and values["surface_tension"] > 0

    window.close()
    app.quit()


def test_metrics_panel_update():
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    panel = window.metrics_panel
    panel.set_metrics(ift=1.2, wo=0.5, volume=3.4, contact_angle=45.0, height=2.0, diameter=4.0)

    assert panel.ift_label.text().startswith("1.2")
    assert panel.volume_label.text().startswith("3.4")

    window.close()
    app.quit()


def test_calibration_box(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2
    from PySide6.QtCore import QPoint

    img = np.zeros((20, 20, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    window.parameter_panel.calibration_mode.setChecked(True)

    class DummyEvent:
        def __init__(self, x, y):
            self._pos = QPoint(x, y)

        def pos(self):
            return self._pos

        def accept(self):
            pass

    window._box_press(DummyEvent(1, 1))
    window._box_move(DummyEvent(10, 10))
    window._box_release(DummyEvent(10, 10))

    assert window.calibration_rect is not None
    x1, y1, x2, y2 = window.calibration_rect
    assert x2 > x1 and y2 > y1

    window.close()
    app.quit()


def test_calibration_line(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2
    from PySide6.QtCore import QPoint

    img = np.zeros((20, 20, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    window.parameter_panel.calibration_mode.setChecked(True)
    window.parameter_panel.manual_toggle.setChecked(True)

    class DummyEvent:
        def __init__(self, x, y):
            self._pos = QPoint(x, y)

        def pos(self):
            return self._pos

        def accept(self):
            pass

    window._line_press(DummyEvent(2, 2))
    window._line_move(DummyEvent(8, 8))
    window._line_release(DummyEvent(8, 8))

    assert window.calibration_line is not None
    x1, y1, x2, y2 = window.calibration_line
    assert x1 != x2 or y1 != y2

    window.close()
    app.quit()


def test_parameter_panel_calibration_controls():
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()

    panel = window.parameter_panel
    assert panel.calibration_method() == "manual"
    panel.manual_toggle.setChecked(False)
    assert panel.calibration_method() == "automatic"
    panel.set_scale_display(5.0)
    assert panel.scale_label.text().startswith("5.0")

    window.close()
    app.quit()


def test_roi_box(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2
    from PySide6.QtCore import QPoint

    img = np.zeros((20, 20, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    window.parameter_panel.roi_mode.setChecked(True)

    class DummyEvent:
        def __init__(self, x, y):
            self._pos = QPoint(x, y)

        def pos(self):
            return self._pos

        def accept(self):
            pass

    window._roi_press(DummyEvent(1, 1))
    window._roi_move(DummyEvent(10, 10))
    window._roi_release(DummyEvent(10, 10))

    assert window.roi_rect is not None
    x1, y1, x2, y2 = window.roi_rect
    assert x2 > x1 and y2 > y1

    window.close()
    app.quit()


def test_process_image_with_roi(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2

    img = np.zeros((20, 30), dtype=np.uint8)
    img[5:15, 10:20] = 255
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    window.roi_rect = (10, 5, 20, 15)
    window.process_image()

    assert window.mask_item is not None
    pixmap = window.mask_item.pixmap()
    assert pixmap.width() == 10
    assert pixmap.height() == 10

    window.close()
    app.quit()

