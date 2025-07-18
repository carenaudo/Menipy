"""Tests for gui module."""

import pytest
import importlib

try:
    from menipy.gui import draw_drop_overlay
except Exception:
    draw_drop_overlay = None

try:
    from menipy.gui import MainWindow
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


def test_save_profiles_option(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib not available")

    import numpy as np
    import cv2
    import os

    img = np.zeros((20, 20), dtype=np.uint8)
    cv2.circle(img, (10, 15), 5, 255, -1)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    cwd = os.getcwd()
    os.chdir(tmp_path)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    window.drop_rect = (5, 10, 15, 19)
    window.px_per_mm_drop = 10.0
    window.pendant_tab.save_profiles_checkbox.setChecked(True)
    window.analyze_drop_image()
    window.close()
    app.quit()
    os.chdir(cwd)

    plot_dir = tmp_path / "plot"
    files = list(plot_dir.glob("*.png"))
    assert len(files) == 2


def test_tab_widget_setup():
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.tabs.count() == 4
    assert window.tabs.tabText(0) == "Calibration"
    assert window.tabs.tabText(1) == "Pendant drop"
    assert window.tabs.tabText(2) == "Contact angle Alt"
    assert window.tabs.tabText(3) == "Detection Test"
    window.close()
    app.quit()


def test_widget_default_sizes():
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    assert window.graphics_view.minimumWidth() == 200
    assert window.graphics_view.minimumHeight() == 200
    assert window.tabs.minimumWidth() == 250
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

    assert 0 < window.graphics_view.scale_factor <= 1.0

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
    before = window.graphics_view.scale_factor
    window.zoom_control.slider.setValue(200)
    after = window.graphics_view.scale_factor
    assert after == pytest.approx(before * 2.0, rel=0.01)

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
    panel.set_metrics(
        ift=1.2,
        wo=0.5,
        volume=3.4,
        contact_angle=45.0,
        height=2.0,
        diameter=4.0,
        mode="sessile",
    )

    assert panel.ift_label.text().startswith("1.2")
    assert panel.volume_label.text().startswith("3.4")

    metrics = panel.values()
    assert metrics["ift"] == pytest.approx(1.2)
    assert metrics["mode"] == "sessile"

    window.close()
    app.quit()


def test_calibration_box(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2
    from PySide6.QtCore import QPoint, QLineF
    from menipy.gui import SubstrateLineItem

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


def test_apex_and_contact_markers(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2

    img = np.zeros((20, 20), dtype=np.uint8)
    img[10:18, 8:12] = 255
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    window.roi_rect = (8, 10, 12, 20)
    window.process_image()

    assert window.apex_item is not None
    assert window.contact_line_item is not None
    line = window.contact_line_item.line()
    assert line.y1() == line.y2() == 10

    window.close()
    app.quit()


def test_calculate_and_draw(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2

    img = np.zeros((20, 20), dtype=np.uint8)
    img[8:18, 8:12] = 255
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    window.process_image()

    window.calculate_parameters()
    assert float(window.metrics_panel.ift_label.text()) >= 0

    window.draw_model()
    assert window.model_item is not None

    window.close()
    app.quit()


def test_save_csv(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2
    import pandas as pd

    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:8, 2:8] = 255
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    window.process_image()
    window.calculate_parameters()

    out_path = tmp_path / "out.csv"
    window.save_csv(out_path)

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert "air_density" in df.columns
    assert "ift" in df.columns
    assert "mode" in df.columns

    window.close()
    app.quit()


def test_draw_drop_overlay_pixmap():
    if QtWidgets is None or draw_drop_overlay is None:
        pytest.skip("PySide6 not available")
    import numpy as np

    contour = np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=float)
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    pix = draw_drop_overlay(
        img,
        contour,
        diameter_line=((5, 10), (15, 10)),
        axis_line=((10, 5), (10, 15)),
        contact_line=((5, 5), (15, 5)),
        apex=(10, 10),
        contact_pts=((5, 5), (15, 5)),
        center_pt=(10, 10),
        center_apex_line=((10, 10), (10, 10)),
        center_contact_line=((10, 10), (10, 5)),
    )
    assert pix.width() == 20 and pix.height() == 20


def test_drop_analysis_workflow(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    import numpy as np
    import cv2

    img = np.zeros((40, 30), dtype=np.uint8)
    cv2.line(img, (14, 5), (14, 35), 0, 2)
    cv2.line(img, (16, 5), (16, 35), 0, 2)
    cv2.circle(img, (15, 30), 8, 255, -1)
    path = tmp_path / "drop.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    window.needle_rect = (12, 5, 18, 35)
    window.detect_needle()
    assert window.px_per_mm_drop > 0
    window.drop_rect = (5, 20, 25, 39)
    window.analyze_drop_image()
    metrics = window.pendant_tab.metrics()
    assert float(metrics["height"]) > 0
    assert window.drop_contour_item is not None
    assert window.drop_rect_item is not None
    window.close()
    app.quit()


def test_drop_regions_saved(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2

    img = np.zeros((10, 10), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    window.needle_rect = (1, 2, 3, 4)
    window.calibration_tab.set_regions(needle=window.needle_rect)
    window.drop_rect = (2, 3, 4, 5)
    window.calibration_tab.set_regions(drop=window.drop_rect)

    assert window.calibration_tab.regions()["needle"] == "1,2,3,4"
    assert window.calibration_tab.regions()["drop"] == "2,3,4,5"

    window.load_image(path)
    assert window.calibration_tab.regions()["needle"] == ""
    assert window.calibration_tab.regions()["drop"] == ""

    window.close()
    app.quit()


def test_substrate_line_updates_metrics(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    import numpy as np
    import cv2
    from PySide6.QtCore import QPoint

    img = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(img, (20, 30), 8, 255, -1)
    path = tmp_path / "drop.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    window.drop_rect = (10, 20, 30, 39)
    window.px_per_mm_drop = 10.0
    window.substrate_line_item = SubstrateLineItem(QLineF(10, 38, 30, 38))
    window.graphics_scene.addItem(window.substrate_line_item)
    window._run_analysis("contact-angle-alt")
    before = window.contact_tab_alt.width_label.text()
    line = window.substrate_line_item.line()
    line.translate(0, -2)
    window.substrate_line_item.setLine(line)
    QtWidgets.QApplication.processEvents()
    after = window.contact_tab_alt.width_label.text()
    assert before == after
    window._run_analysis("contact-angle-alt")
    after = window.contact_tab_alt.width_label.text()
    window.close()
    app.quit()
    assert before != after


def test_contact_tab_draw_button(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    import numpy as np
    import cv2
    from unittest.mock import patch
    from PySide6.QtCore import QLineF
    from menipy.gui import SubstrateLineItem

    img = np.zeros((20, 20), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    assert window.contact_tab_alt.substrate_button is not None
    window.contact_tab_alt.substrate_button.click()
    assert window.draw_substrate_action.isChecked()
    with patch("PySide6.QtWidgets.QMessageBox.warning") as warn:
        window._run_analysis("contact-angle-alt")
        assert warn.called
    window.substrate_line_item = SubstrateLineItem(QLineF(1, 10, 18, 10))
    window.graphics_scene.addItem(window.substrate_line_item)
    with patch("PySide6.QtWidgets.QMessageBox.warning") as warn:
        window._run_analysis("contact-angle-alt")
        assert not warn.called
    window.close()
    app.quit()


def test_contact_tab_detect_button(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    import numpy as np
    import cv2

    img = np.full((40, 40), 255, dtype=np.uint8)
    cv2.line(img, (2, 30), (38, 30), 0, 2)
    cv2.circle(img, (20, 24), 6, 0, -1)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    window.drop_rect = (0, 10, 39, 39)
    window.contact_tab_alt.detect_substrate_button.click()
    assert window.substrate_line_item is not None
    line = window.substrate_line_item.line()
    ang = np.degrees(np.arctan2(line.y2() - line.y1(), line.x2() - line.x1()))
    if ang > 90.0:
        ang -= 180.0
    elif ang < -90.0:
        ang += 180.0
    assert abs(ang) <= 5.0
    window.close()
    app.quit()


def test_contact_tab_side_button(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    import numpy as np
    import cv2
    from unittest.mock import patch
    from PySide6.QtCore import QLineF
    from menipy.gui import SubstrateLineItem

    img = np.zeros((20, 20), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    assert window.contact_tab_alt.side_button is not None
    with patch("PySide6.QtWidgets.QMessageBox.information") as info:
        window.contact_tab_alt.side_button.click()
        assert info.called
    window.substrate_line_item = SubstrateLineItem(QLineF(1, 10, 18, 10))
    window.graphics_scene.addItem(window.substrate_line_item)
    with patch("PySide6.QtWidgets.QMessageBox.information") as info:
        window.contact_tab_alt.side_button.click()
        assert not info.called
    window.close()
    app.quit()


def test_clear_analysis_button(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    import numpy as np
    import cv2

    img = np.zeros((20, 20, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    window.filter_slider.setValue(2)
    window.apply_filter()
    window.process_image()
    assert window.mask_item is not None
    window.clear_analysis_button.click()
    assert window.mask_item is None
    assert np.array_equal(window.image, window.original_image)
    window.close()
    app.quit()


def test_clear_analysis_resets_state(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    import numpy as np
    import cv2
    from PySide6.QtCore import QPoint

    class DummyEvent:
        def __init__(self, x, y):
            self._pos = QPoint(x, y)

        def pos(self):
            return self._pos

        def accept(self):
            pass

    img = np.zeros((20, 20, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    window.set_needle_mode(True)
    window._needle_press(DummyEvent(1, 1))
    window._needle_move(DummyEvent(5, 5))
    window._needle_release(DummyEvent(5, 5))
    window.parameter_panel.calibration_mode.setChecked(True)
    window.parameter_panel.manual_toggle.setChecked(True)
    window._line_press(DummyEvent(0, 0))
    window._line_move(DummyEvent(10, 0))
    window._line_release(DummyEvent(10, 0))

    assert window.needle_rect_item is not None
    assert window.calibration_line_item is not None

    window.clear_analysis_button.click()

    assert window.needle_rect_item is None
    assert window.calibration_line_item is None
    assert window.roi_rect_item is None
    assert window.mask_item is None
    assert np.array_equal(window.image, window.original_image)

    window.set_needle_mode(True)
    window._needle_press(DummyEvent(2, 2))
    window._needle_move(DummyEvent(6, 6))
    window._needle_release(DummyEvent(6, 6))
    assert window.needle_rect is not None
    window.close()
    app.quit()


def test_clear_analysis_resets_metrics(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")
    import numpy as np
    import cv2

    img = np.zeros((20, 20, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)

    window.pendant_tab.set_metrics(height=1.0)
    window.contact_tab_alt.set_metrics(diameter=2.0)
    window.metrics_panel.set_metrics(ift=3.0, contact_angle=40.0)

    window.clear_analysis_button.click()

    assert window.metrics_panel.ift_label.text() == "0.0"
    assert window.pendant_tab.height_label.text() == "0.0000"
    assert window.contact_tab_alt.diameter_label.text() == "0.0000"
    window.close()
    app.quit()


def test_contact_angle_alt_metrics(tmp_path):
    if QtWidgets is None:
        pytest.skip("PySide6 not available")

    import numpy as np
    import cv2
    from PySide6.QtCore import QLineF
    from menipy.gui import SubstrateLineItem

    img = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(img, (20, 30), 8, 255, -1)
    path = tmp_path / "drop.png"
    cv2.imwrite(str(path), img)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow()
    window.load_image(path)
    window.drop_rect = (10, 20, 30, 39)
    window.px_per_mm_drop = 10.0
    window.substrate_line_item = SubstrateLineItem(QLineF(10, 38, 30, 38))
    window.graphics_scene.addItem(window.substrate_line_item)
    window._run_analysis("contact-angle-alt")

    txt1 = window.contact_tab_alt.angle_p1_label.text()
    txt2 = window.contact_tab_alt.angle_p2_label.text()
    assert float(txt1) > 0
    assert float(txt2) > 0

    window.close()
    app.quit()
