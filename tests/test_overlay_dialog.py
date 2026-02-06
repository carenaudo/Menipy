"""Tests for test overlay dialog.

Unit tests."""


import sys
import pytest

from PySide6.QtWidgets import QApplication

from menipy.gui.dialogs.overlay_config_dialog import OverlayConfigDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_get_set_config_roundtrip(qapp):
    dlg = OverlayConfigDialog()
    cfg = dlg.get_config()
    assert isinstance(cfg, dict)
    # modify a value and set it back
    cfg["contour_thickness"] = 5
    dlg.set_config(cfg)
    new_cfg = dlg.get_config()
    assert new_cfg["contour_thickness"] == 5


def test_signals_present(qapp):
    dlg = OverlayConfigDialog()
    # Ensure the signal attributes exist and are callable
    assert hasattr(dlg, "previewRequested")
    assert hasattr(dlg, "configApplied")
