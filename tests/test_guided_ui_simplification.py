from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import MethodType
from unittest.mock import Mock

from PySide6.QtCore import QFile
from PySide6.QtGui import QFontDatabase
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMainWindow

from menipy.gui.controllers.setup_panel_controller import SetupPanelController
from menipy.gui.views.main_window import (
    MainWindow,
    _workbench_root_sizes,
    _workbench_vertical_sizes,
)


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
    assert not controller.advancedToggleBtn.isCheckable()
    assert not controller.advancedToggleBtn.isChecked()
    assert controller.sopGroup is not None
    assert not controller.sopGroup.isVisible()
    assert controller.stepsGroup is not None
    assert not controller.stepsGroup.isVisible()


def test_guided_setup_advanced_button_is_dialog_command(qtbot):
    controller = _controller(qtbot)
    requested = Mock()
    controller.advanced_requested.connect(requested)

    controller.advancedToggleBtn.click()

    requested.assert_called_once()
    assert not controller.sopGroup.isVisible()
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
    assert controller.autoCalibrateBtn.text() == "Calibrate"
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


def test_guided_setup_keeps_all_supported_analysis_buttons(qtbot):
    controller = _controller(qtbot)

    buttons = {
        controller.sessileBtn.toolTip(): "sessile",
        controller.pendantBtn.toolTip(): "pendant",
        controller.oscillatingBtn.toolTip(): "oscillating",
        controller.capillaryBtn.toolTip(): "capillary_rise",
        controller.captiveBtn.toolTip(): "captive_bubble",
    }

    assert buttons == {
        "Sessile": "sessile",
        "Pendant": "pendant",
        "Osc.": "oscillating",
        "Capillary": "capillary_rise",
        "Captive": "captive_bubble",
    }


def test_guided_setup_analysis_buttons_show_text_only_when_selected(qtbot):
    controller = _controller(qtbot)

    assert controller.sessileBtn.isChecked()
    assert controller.sessileBtn.text() == "Sessile"
    assert controller.sessileBtn.icon().isNull()
    assert controller.pendantBtn.text() == ""
    assert not controller.pendantBtn.icon().isNull()

    controller.pendantBtn.click()

    assert controller.sessileBtn.text() == ""
    assert not controller.sessileBtn.icon().isNull()
    assert controller.pendantBtn.text() == "Pendant"
    assert controller.pendantBtn.icon().isNull()


def test_guided_setup_pipeline_specific_fields_follow_selection(qtbot):
    controller = _controller(qtbot)

    controller.pendantBtn.click()
    assert controller.pipelineSettingsGroup.title() == "Pendant Settings"
    assert not controller.pendantNeedleIdSpin.isHidden()
    assert controller.sessileBaselineModeCombo.isHidden()

    controller.oscillatingBtn.click()
    assert controller.pipelineSettingsGroup.title() == "Oscillating Settings"
    assert not controller.oscillatingFrequencySpin.isHidden()
    assert not controller.oscillatingAmplitudeSpin.isHidden()
    assert controller.pendantNeedleIdSpin.isHidden()

    controller.capillaryBtn.click()
    assert controller.pipelineSettingsGroup.title() == "Capillary Settings"
    assert not controller.capillaryTubeDiameterSpin.isHidden()
    assert not controller.capillaryContactAngleSpin.isHidden()

    controller.captiveBtn.click()
    assert controller.pipelineSettingsGroup.title() == "Captive Bubble Settings"
    assert not controller.captiveDetectionValue.isHidden()


def test_guided_setup_collects_analysis_params(qtbot):
    controller = _controller(qtbot)

    controller.capillaryBtn.click()
    controller.capillaryTubeDiameterSpin.setValue(1.5)
    params = controller.gather_run_params()

    assert params["calibration_params"]["g"] == controller.gravitySpin.value()
    assert params["analysis_params"]["pipeline"] == "capillary_rise"
    assert params["analysis_params"]["tube_diameter_mm"] == 1.5


def test_workbench_root_sizes_migrates_legacy_result_heavy_layout():
    sizes = _workbench_root_sizes(1500, [240, 500, 760])

    assert sizes[1] > sizes[0]


def test_workbench_sizes_preserve_preview_dominant_saved_layout():
    assert _workbench_root_sizes(1500, [260, 1240]) == [260, 1240]
    assert _workbench_vertical_sizes(900, [640, 260]) == [640, 260]


def test_theme_font_resolver_uses_available_candidate(qtbot, monkeypatch):
    from menipy.gui import theme

    available = QFontDatabase.families()
    assert available
    fallback_family = available[0]
    monkeypatch.setattr(
        theme,
        "FONT_FAMILY_CANDIDATES",
        ("Definitely Missing Menipy Font", fallback_family),
    )

    assert theme.resolve_font_family() == fallback_family


def test_workbench_layout_xml_places_preview_above_results():
    tree = ET.parse("src/menipy/gui/views/main_window_split.ui")
    root = tree.getroot()

    splitter = root.find(".//widget[@name='workbenchSplitter']")
    assert splitter is not None
    child_names = [
        child.attrib.get("name") for child in splitter if child.tag == "widget"
    ]
    assert child_names[:2] == ["previewResultsSplitter", "inspectTabs"]

    preview_splitter = root.find(".//widget[@name='previewResultsSplitter']")
    assert preview_splitter is not None
    preview_child_names = [
        child.attrib.get("name") for child in preview_splitter if child.tag == "widget"
    ]
    assert preview_child_names[:2] == ["previewHost", "keyResultsHost"]


def test_workflow_bar_xml_removes_duplicate_calibrate_and_run_buttons():
    tree = ET.parse("src/menipy/gui/views/main_window_split.ui")
    root = tree.getroot()

    assert root.find(".//widget[@name='workflowAutoCalibrateBtn']") is None
    assert root.find(".//widget[@name='actionRunBtn']") is None
    assert root.find(".//widget[@name='workflowAdvancedBtn']") is None


def test_workflow_bar_xml_hosts_analysis_source_and_right_toggles():
    tree = ET.parse("src/menipy/gui/views/main_window_split.ui")
    root = tree.getroot()

    assert root.find(".//widget[@name='workflowAnalysisHost']") is not None
    assert root.find(".//layout[@name='workflowAnalysisLayout']") is not None
    assert root.find(".//widget[@name='workflowSourceHost']") is not None
    assert root.find(".//layout[@name='workflowSourceLayout']") is not None
    assert root.find(".//widget[@name='workflowPanelToggleHost']") is not None

    bar_layout = root.find(".//layout[@name='workflowBarLayout']")
    assert bar_layout is not None
    widget_names = [
        widget.attrib.get("name")
        for item in bar_layout.findall("item")
        for widget in item.findall("widget")
    ]
    assert widget_names[:2] == ["workflowAnalysisHost", "workflowSourceHost"]
    assert widget_names[-1] == "workflowPanelToggleHost"


def test_menu_xml_order_and_config_actions():
    tree = ET.parse("src/menipy/gui/views/main_window_split.ui")
    root = tree.getroot()
    menubar = root.find(".//widget[@name='menubar']")
    assert menubar is not None
    menu_order = [action.attrib["name"] for action in menubar.findall("addaction")]
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
        "menuUnits",
        "actionConfigAcquisition",
        "actionConfigPreprocessing",
        "actionConfigEdgeDetection",
        "actionConfigGeometry",
        "actionConfigPhysics",
    ]

    view = root.find(".//widget[@name='menuView']")
    assert view is not None
    view_actions = [
        action.attrib["name"]
        for action in view.findall("addaction")
        if action.attrib["name"] != "separator"
    ]
    assert view_actions == [
        "menuFocus",
        "menuPanels",
        "actionResetLayout",
        "actionFitPreview",
    ]

    focus = root.find(".//widget[@name='menuFocus']")
    assert focus is not None
    focus_actions = [
        action.attrib["name"]
        for action in focus.findall("addaction")
        if action.attrib["name"] != "separator"
    ]
    assert focus_actions == [
        "actionFocusMeasure",
        "actionFocusScience",
        "actionFocusAnalysis",
    ]

    panels = root.find(".//widget[@name='menuPanels']")
    assert panels is not None
    panel_actions = [
        action.attrib["name"]
        for action in panels.findall("addaction")
        if action.attrib["name"] != "separator"
    ]
    assert panel_actions == [
        "actionToggleSetup",
        "actionToggleResultsTable",
        "actionToggleKeyResults",
    ]


class _FakeHost:
    def __init__(self, visible=True):
        self.visible = visible

    def setVisible(self, visible):
        self.visible = bool(visible)

    def isVisible(self):
        return self.visible

    def isHidden(self):
        return not self.visible


class _FakeSplitter:
    def __init__(self, sizes):
        self._sizes = list(sizes)

    def sizes(self):
        return list(self._sizes)

    def setSizes(self, sizes):
        self._sizes = list(sizes)


class _FakeTabs(_FakeHost):
    def __init__(self, results_tab, residuals_tab):
        super().__init__(True)
        self._tabs = [results_tab, residuals_tab]
        self.current = results_tab

    def indexOf(self, tab):
        return self._tabs.index(tab) if tab in self._tabs else -1

    def setCurrentIndex(self, index):
        self.current = self._tabs[index]

    def currentWidget(self):
        return self.current


class _FakeControl:
    def __init__(self):
        self.checked = None

    def blockSignals(self, _blocked):
        pass

    def setChecked(self, checked):
        self.checked = bool(checked)


class _FakeResultsPanel:
    def __init__(self):
        self.settings = Mock(compare_methods_visible=False, diagnostics_visible=False)
        self.key_results_shown = False

    def show_key_results(self):
        self.key_results_shown = True
        self.settings.compare_methods_visible = False
        self.settings.diagnostics_visible = False

    def set_compare_methods_visible(self, visible, *, refresh=True):
        self.settings.compare_methods_visible = bool(visible)

    def set_diagnostics_visible(self, visible, *, refresh=True):
        self.settings.diagnostics_visible = bool(visible)

    def has_residuals_for_current_selection(self):
        return False


def _focus_window():
    window = Mock()
    window.width.return_value = 1200
    window.height.return_value = 800
    window.rootSplitter = _FakeSplitter([320, 880])
    window.workbenchSplitter = _FakeSplitter([560, 240])
    window.previewResultsSplitter = _FakeSplitter([900, 320])
    window.setupHost = _FakeHost(True)
    window.keyResultsHost = _FakeHost(True)
    window.resultsTab = object()
    window.residualsTab = object()
    window.inspectTabs = _FakeTabs(window.resultsTab, window.residualsTab)
    window.results_panel_ctrl = _FakeResultsPanel()
    window.toggleSetupBtn = _FakeControl()
    window.toggleInspectBtn = _FakeControl()
    window.toggleKeyResultsBtn = _FakeControl()
    window.actionToggleSetup = _FakeControl()
    window.actionToggleResultsTable = _FakeControl()
    window.actionToggleKeyResults = _FakeControl()
    window._splitter_sizes_full = None
    window._workbench_splitter_sizes_full = None
    window._preview_results_sizes_full = None
    for name in (
        "_apply_focus_preset",
        "_apply_focus_sizes",
        "_select_inspect_tab",
        "_set_panel_visibility",
        "_sync_layout_buttons",
    ):
        setattr(window, name, MethodType(getattr(MainWindow, name), window))
    return window


def test_focus_presets_apply_workbench_modes():
    window = _focus_window()

    window._apply_focus_preset("measure")
    assert not window.setupHost.isHidden()
    assert window.inspectTabs.isHidden()
    assert not window.keyResultsHost.isHidden()

    window._apply_focus_preset("science")
    assert window.setupHost.isHidden()
    assert not window.inspectTabs.isHidden()
    assert not window.keyResultsHost.isHidden()
    assert window.inspectTabs.currentWidget() in (
        window.resultsTab,
        window.residualsTab,
    )
    assert window.results_panel_ctrl.settings.diagnostics_visible
    assert not window.results_panel_ctrl.settings.compare_methods_visible

    window._apply_focus_preset("analysis")
    assert window.setupHost.isHidden()
    assert not window.inspectTabs.isHidden()
    assert window.keyResultsHost.isHidden()
    assert window.inspectTabs.currentWidget() is window.resultsTab
    assert window.results_panel_ctrl.settings.compare_methods_visible
    assert not window.results_panel_ctrl.settings.diagnostics_visible
