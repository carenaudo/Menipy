"""Tests for test setup panel.

Unit tests."""

import pytest
from menipy.gui.main_window import MainWindow
from menipy.gui.controllers.setup_panel_controller import SetupPanelController
from menipy.gui.views.image_view import DRAW_POINT, DRAW_LINE, DRAW_RECT
from unittest.mock import Mock
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListView


@pytest.fixture
def main_window(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    yield window
    if getattr(window, "main_controller", None):
        window.main_controller.shutdown()


@pytest.fixture
def setup_panel_controller(main_window) -> SetupPanelController:
    controller = main_window.setup_panel_ctrl
    for signal_name in (
        "browse_requested",
        "browse_batch_requested",
        "preview_requested",
        "run_all_requested",
        "play_stage_requested",
        "config_stage_requested",
        "source_mode_changed",
        "auto_calibrate_requested",
    ):
        signal = getattr(controller, signal_name)
        try:
            signal.disconnect()
        except RuntimeError:
            pass
    return controller


def test_radio_button_toggles_mode(qtbot, setup_panel_controller: SetupPanelController):
    mock_signal = Mock()
    setup_panel_controller.source_mode_changed.connect(mock_signal)

    # Test single mode
    if setup_panel_controller.singleModeRadio:
        qtbot.mouseClick(setup_panel_controller.singleModeRadio, Qt.LeftButton)
        assert (
            setup_panel_controller.current_mode() == setup_panel_controller.MODE_SINGLE
        )
        mock_signal.assert_called_with(setup_panel_controller.MODE_SINGLE)
        mock_signal.reset_mock()

    # Test batch mode
    if setup_panel_controller.batchModeRadio:
        qtbot.mouseClick(setup_panel_controller.batchModeRadio, Qt.LeftButton)
        assert (
            setup_panel_controller.current_mode() == setup_panel_controller.MODE_BATCH
        )
        mock_signal.assert_called_with(setup_panel_controller.MODE_BATCH)
        mock_signal.reset_mock()

    # Test camera mode
    if setup_panel_controller.cameraModeRadio:
        qtbot.mouseClick(setup_panel_controller.cameraModeRadio, Qt.LeftButton)
        assert (
            setup_panel_controller.current_mode() == setup_panel_controller.MODE_CAMERA
        )
        mock_signal.assert_called_with(setup_panel_controller.MODE_CAMERA)
        mock_signal.reset_mock()


def test_browse_button_emits_signal(
    qtbot, setup_panel_controller: SetupPanelController
):
    mock_signal = Mock()
    setup_panel_controller.browse_requested.connect(mock_signal)
    if setup_panel_controller.browseBtn:
        qtbot.mouseClick(setup_panel_controller.browseBtn, Qt.LeftButton)
        mock_signal.assert_called_once()


def test_batch_browse_button_emits_signal(
    qtbot, setup_panel_controller: SetupPanelController
):
    mock_signal = Mock()
    setup_panel_controller.browse_batch_requested.connect(mock_signal)
    if setup_panel_controller.batchBrowseBtn:
        qtbot.mouseClick(setup_panel_controller.batchBrowseBtn, Qt.LeftButton)
        mock_signal.assert_called_once()


def test_preview_button_emits_signal(
    qtbot, setup_panel_controller: SetupPanelController
):
    mock_signal = Mock()
    setup_panel_controller.preview_requested.connect(mock_signal)
    if setup_panel_controller.previewBtn:
        qtbot.mouseClick(setup_panel_controller.previewBtn, Qt.LeftButton)
        mock_signal.assert_called_once()


def test_draw_mode_buttons_emit_signal(
    qtbot, setup_panel_controller: SetupPanelController
):
    mock_signal = Mock()
    setup_panel_controller.draw_mode_requested.connect(mock_signal)

    if setup_panel_controller.drawPointBtn:
        qtbot.mouseClick(setup_panel_controller.drawPointBtn, Qt.LeftButton)
        mock_signal.assert_called_with(DRAW_POINT)
        mock_signal.reset_mock()

    if setup_panel_controller.drawLineBtn:
        qtbot.mouseClick(setup_panel_controller.drawLineBtn, Qt.LeftButton)
        mock_signal.assert_called_with(DRAW_LINE)
        mock_signal.reset_mock()

    if setup_panel_controller.drawRectBtn:
        qtbot.mouseClick(setup_panel_controller.drawRectBtn, Qt.LeftButton)
        mock_signal.assert_called_with(DRAW_RECT)
        mock_signal.reset_mock()


def test_clear_overlay_button_emits_signal(
    qtbot, setup_panel_controller: SetupPanelController
):
    mock_signal = Mock()
    setup_panel_controller.clear_overlays_requested.connect(mock_signal)
    if setup_panel_controller.clearOverlayBtn:
        qtbot.mouseClick(setup_panel_controller.clearOverlayBtn, Qt.LeftButton)
        mock_signal.assert_called_once()


def test_run_all_button_emits_signal(
    qtbot, setup_panel_controller: SetupPanelController
):
    mock_signal = Mock()
    setup_panel_controller.run_all_requested.connect(mock_signal)
    if setup_panel_controller.runAllBtn:
        qtbot.mouseClick(setup_panel_controller.runAllBtn, Qt.LeftButton)
        mock_signal.assert_called_once()


def test_pipeline_combo_changes_pipeline(
    qtbot, setup_panel_controller: SetupPanelController
):
    mock_signal = Mock()
    setup_panel_controller.pipeline_changed.connect(mock_signal)
    combo = setup_panel_controller.testCombo or setup_panel_controller.pipelineCombo
    if combo and combo.count() > 1:
        initial_text = combo.currentText()
        combo.setCurrentIndex(1)
        qtbot.wait_signal(combo.currentTextChanged)
        assert combo.currentText() != initial_text
        mock_signal.assert_called_once_with(
            combo.currentText().lower().replace(" ", "_")
        )


def test_image_path_edit_refreshes_source_items(
    qtbot, setup_panel_controller: SetupPanelController
):
    if setup_panel_controller.imagePathEdit:
        original_refresh_method = setup_panel_controller._refresh_source_items
        setup_panel_controller._refresh_source_items = Mock()
        setup_panel_controller.imagePathEdit.setText("test_path.png")
        setup_panel_controller.imagePathEdit.textChanged.emit(
            "test_path.png"
        )  # Explicitly emit signal
        qtbot.wait(100)  # Give some time for signal to propagate
        setup_panel_controller._refresh_source_items.assert_called_once()
        setup_panel_controller._refresh_source_items = (
            original_refresh_method  # Restore original
        )


def test_batch_path_edit_refreshes_source_items(
    qtbot, setup_panel_controller: SetupPanelController
):
    if setup_panel_controller.batchPathEdit:
        original_refresh_method = setup_panel_controller._refresh_source_items
        setup_panel_controller._refresh_source_items = Mock()
        setup_panel_controller.batchPathEdit.setText("test_batch_folder")
        setup_panel_controller.batchPathEdit.textChanged.emit(
            "test_batch_folder"
        )  # Explicitly emit signal
        qtbot.wait(100)  # Give some time for signal to propagate
        setup_panel_controller._refresh_source_items.assert_called_once()
        setup_panel_controller._refresh_source_items = (
            original_refresh_method  # Restore original
        )


def test_add_sop_button_calls_sop_controller(
    qtbot, setup_panel_controller: SetupPanelController
):
    if setup_panel_controller.addSopBtn:
        setup_panel_controller.sop_ctrl.on_add_sop = Mock()
        qtbot.mouseClick(setup_panel_controller.addSopBtn, Qt.LeftButton)
        setup_panel_controller.sop_ctrl.on_add_sop.assert_called_once()


def test_steps_list_uses_vertical_layout(setup_panel_controller: SetupPanelController):
    steps_list = setup_panel_controller.stepsList
    assert steps_list is not None
    assert steps_list.flow() == QListView.TopToBottom
    assert not steps_list.isWrapping()
    assert steps_list.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


def test_excluded_step_stays_readable_but_is_not_collected(
    setup_panel_controller: SetupPanelController,
):
    widgets = setup_panel_controller.sop_ctrl._step_widgets
    assert widgets
    first = widgets[0]

    first.set_included(False)

    assert first.isEnabled()
    assert not first.is_included()
    assert first.step_name not in setup_panel_controller.collect_included_stages()


def test_included_step_controls_emit_signals(qtbot, setup_panel_controller):
    widgets = setup_panel_controller.sop_ctrl._step_widgets
    assert widgets
    first = widgets[0]
    first.set_included(True)
    handler = Mock()
    first.playClicked.connect(handler)

    qtbot.mouseClick(first.playBtn, Qt.LeftButton)

    handler.assert_called_once_with(first.step_name)
