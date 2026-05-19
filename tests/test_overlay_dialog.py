"""Tests for test overlay dialog.

Unit tests."""


import sys
import pytest

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from menipy.gui.dialogs.overlay_config_dialog import OverlayConfigDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_get_set_config_roundtrip(qapp):
    dlg = OverlayConfigDialog()
    cfg = dict(OverlayConfigDialog.DEFAULTS)
    assert isinstance(cfg, dict)
    # modify a value and set it back
    cfg["contour_thickness"] = 5
    cfg["fit_color"] = "#123456"
    cfg["fit_thickness"] = 4
    cfg["stroke_scale_mode"] = "image"
    dlg.set_config(cfg)
    new_cfg = dlg.get_config()
    assert new_cfg["contour_thickness"] == 5
    assert new_cfg["fit_color"] == "#123456"
    assert new_cfg["fit_thickness"] == 4
    assert new_cfg["stroke_scale_mode"] == "image"
    assert set(OverlayConfigDialog.DEFAULTS).issubset(new_cfg)


def test_signals_present(qapp):
    dlg = OverlayConfigDialog()
    # Ensure the signal attributes exist and are callable
    assert hasattr(dlg, "previewRequested")
    assert hasattr(dlg, "configApplied")


def test_layer_rows_exist(qapp):
    dlg = OverlayConfigDialog()

    assert set(dlg._layer_controls) == {
        "contour",
        "fit",
        "axes",
        "baseline",
        "markers",
        "points",
    }
    assert dlg._layer_controls["contour"]["dashed"] is dlg.contour_dashed
    assert dlg._layer_controls["points"]["thickness"] is dlg.point_radius


def test_dialog_has_practical_size_and_generated_preview(qapp):
    dlg = OverlayConfigDialog()

    assert dlg.minimumWidth() >= 680
    assert dlg.minimumHeight() >= 520
    pixmap = dlg.preview_label.pixmap()
    assert pixmap is not None
    assert not pixmap.isNull()
    assert pixmap.width() <= 360
    assert pixmap.height() <= 260


def test_layer_color_helpers_update_only_requested_layer(qapp):
    dlg = OverlayConfigDialog()

    dlg._set_layer_color("fit", QColor("#123456"))

    assert dlg.get_config()["fit_color"] == "#123456"
    assert dlg.get_config()["contour_color"] != "#123456"
