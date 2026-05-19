from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from PySide6.QtCore import QFile, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMainWindow

from menipy.gui.controllers.setup_panel_controller import SetupPanelController


class DummySettings:
    selected_pipeline = "sessile"
    last_image_path = None
    unit_system = "SI"
    advanced_ui_visible = False

    def save(self):
        pass


def _load_setup_panel():
    path = Path("src/menipy/gui/views/setup_panel.ui").resolve()
    file = QFile(str(path))
    assert file.open(QFile.ReadOnly)
    try:
        return QUiLoader().load(file)
    finally:
        file.close()


def _controller(qtbot):
    window = QMainWindow()
    panel = _load_setup_panel()
    qtbot.addWidget(window)
    qtbot.addWidget(panel)
    panel.show()
    controller = SetupPanelController(
        window=window,
        panel=panel,
        settings=DummySettings(),
        sops=None,
        stage_order=["acquisition", "preprocessing", "outputs"],
        step_item_cls=None,
        pipeline_keys=[
            "sessile",
            "pendant",
            "oscillating",
            "capillary_rise",
            "captive_bubble",
        ],
    )
    return controller


def test_guided_setup_defaults_collapse_advanced_controls(qtbot):
    controller = _controller(qtbot)

    assert controller.advancedToggleBtn is not None
    assert not controller.advancedToggleBtn.isChecked()
    assert controller.sopGroup is not None
    assert not controller.sopGroup.isVisible()
    assert controller.stepsGroup is not None
    assert not controller.stepsGroup.isVisible()


def test_guided_setup_shows_only_active_source_controls(qtbot):
    controller = _controller(qtbot)

    assert not controller.imagePathEdit.isHidden()
    assert controller.batchPathEdit.isHidden()
    assert controller.framesSpin.isHidden()

    qtbot.mouseClick(controller.batchModeRadio, Qt.LeftButton)
    assert controller.imagePathEdit.isHidden()
    assert not controller.batchPathEdit.isHidden()
    assert not controller.sourceIdCombo.isHidden()

    qtbot.mouseClick(controller.cameraModeRadio, Qt.LeftButton)
    assert controller.imagePathEdit.isHidden()
    assert controller.batchPathEdit.isHidden()
    assert not controller.sourceIdCombo.isHidden()
    assert not controller.framesSpin.isHidden()


def test_guided_setup_keeps_primary_signals_available(qtbot):
    controller = _controller(qtbot)
    browse = Mock()
    preview = Mock()
    calibrate = Mock()
    run = Mock()
    controller.browse_requested.connect(browse)
    controller.preview_requested.connect(preview)
    controller.auto_calibrate_requested.connect(calibrate)
    controller.run_all_requested.connect(run)

    qtbot.mouseClick(controller.browseBtn, Qt.LeftButton)
    qtbot.mouseClick(controller.previewBtn, Qt.LeftButton)
    qtbot.mouseClick(controller.autoCalibrateBtn, Qt.LeftButton)
    qtbot.mouseClick(controller.runAllBtn, Qt.LeftButton)

    browse.assert_called_once()
    preview.assert_called_once()
    calibrate.assert_called_once()
    run.assert_called_once()


def test_guided_setup_hides_substrate_angle_outside_sessile(qtbot):
    controller = _controller(qtbot)

    assert not controller.substrateAngleSpin.isHidden()
    qtbot.mouseClick(controller.pendantBtn, Qt.LeftButton)
    assert controller.substrateAngleSpin.isHidden()
