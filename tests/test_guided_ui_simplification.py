from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock
import xml.etree.ElementTree as ET

from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMainWindow

from menipy.gui.controllers.setup_panel_controller import SetupPanelController
from menipy.gui.main_window import _workbench_root_sizes, _workbench_vertical_sizes


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

    controller.batchModeRadio.click()
    assert controller.imagePathEdit.isHidden()
    assert not controller.batchPathEdit.isHidden()
    assert not controller.sourceIdCombo.isHidden()

    controller.cameraModeRadio.click()
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

    controller.browseBtn.click()
    controller.previewBtn.click()
    controller.autoCalibrateBtn.click()
    controller.runAllBtn.click()

    browse.assert_called_once()
    preview.assert_called_once()
    calibrate.assert_called_once()
    run.assert_called_once()


def test_guided_setup_hides_substrate_angle_outside_sessile(qtbot):
    controller = _controller(qtbot)

    assert not controller.substrateAngleSpin.isHidden()
    controller.pendantBtn.click()
    assert controller.substrateAngleSpin.isHidden()


def test_workbench_root_sizes_migrates_legacy_result_heavy_layout():
    sizes = _workbench_root_sizes(1500, [240, 500, 760])

    assert sizes[1] > sizes[0]


def test_workbench_sizes_preserve_preview_dominant_saved_layout():
    assert _workbench_root_sizes(1500, [260, 1240]) == [260, 1240]
    assert _workbench_vertical_sizes(900, [640, 260]) == [640, 260]


def test_workbench_layout_xml_places_preview_above_results():
    tree = ET.parse("src/menipy/gui/views/main_window_split.ui")
    root = tree.getroot()

    splitter = root.find(".//widget[@name='workbenchSplitter']")
    assert splitter is not None
    child_names = [
        child.attrib.get("name")
        for child in splitter
        if child.tag == "widget"
    ]
    assert child_names[:2] == ["previewHost", "inspectTabs"]


def test_menu_xml_order_and_config_actions():
    tree = ET.parse("src/menipy/gui/views/main_window_split.ui")
    root = tree.getroot()
    menubar = root.find(".//widget[@name='menubar']")
    assert menubar is not None
    menu_order = [
        action.attrib["name"]
        for action in menubar.findall("addaction")
    ]
    assert menu_order == [
        "menuFile",
        "menuConfig",
        "menuView",
        "menuRun",
        "menuPlugins",
        "menuHelp",
    ]

    config = root.find(".//widget[@name='menuConfig']")
    assert config is not None
    config_actions = [
        action.attrib["name"]
        for action in config.findall("addaction")
        if action.attrib["name"] != "separator"
    ]
    assert config_actions == [
        "actionConfigOverlay",
        "actionConfigMarkers",
        "actionConfigPipeline",
        "actionConfigPreprocessing",
        "actionConfigEdgeDetection",
        "actionConfigGeometry",
        "actionConfigPhysics",
        "actionConfigAcquisition",
    ]
