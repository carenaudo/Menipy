"""Tests for test setup panel.

Unit tests."""

import pytest
from menipy.gui.views.main_window import MainWindow
from menipy.gui.controllers.setup_panel_controller import SetupPanelController
from menipy.gui.services.camera_service import CameraDevice
from menipy.gui.views.image_view import DRAW_POINT, DRAW_LINE, DRAW_RECT
from menipy.models.context import Context
from unittest.mock import Mock
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QListView, QSizePolicy


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


def test_setup_action_stack_order_and_labels(setup_panel_controller):
    layout = setup_panel_controller.panel.layout()
    assert layout is not None

    action_indices = [
        layout.indexOf(setup_panel_controller.autoCalibrateBtn),
        layout.indexOf(setup_panel_controller.runAllBtn),
        layout.indexOf(setup_panel_controller.advancedToggleBtn),
    ]

    assert all(index >= 0 for index in action_indices)
    assert action_indices == sorted(action_indices)
    assert setup_panel_controller.autoCalibrateBtn.text() == "Calibrate"
    assert setup_panel_controller.runAllBtn.text() == "Run Analysis"
    assert setup_panel_controller.advancedToggleBtn.text() == "Advanced"
    assert "#2563EB" not in setup_panel_controller.advancedToggleBtn.styleSheet()
    assert (
        setup_panel_controller.advancedToggleBtn.sizePolicy().horizontalPolicy()
        == QSizePolicy.Policy.Expanding
    )


def test_top_workflow_calibrate_and_run_buttons_removed(main_window):
    assert not hasattr(main_window, "workflowAutoCalibrateBtn")
    assert not hasattr(main_window, "actionRunBtn")
    assert not hasattr(main_window, "workflowAdvancedBtn")


def test_science_focus_controls_pipeline_test_button(main_window, qtbot):
    main_window.settings.save = lambda: None
    main_window.results_panel_ctrl.settings.save = lambda: None
    assert main_window.workflowTestPipelineBtn.isHidden()

    main_window.actionFocusScience.trigger()
    qtbot.wait(20)

    assert not main_window.workflowTestPipelineBtn.isHidden()

    main_window.actionFocusMeasure.trigger()
    qtbot.wait(20)

    assert main_window.workflowTestPipelineBtn.isHidden()
    assert not main_window.workflowTestPipelineBtn.isChecked()


def test_pipeline_test_button_swaps_left_rail(main_window, qtbot):
    main_window.settings.save = lambda: None
    main_window.results_panel_ctrl.settings.save = lambda: None
    main_window.actionFocusScience.trigger()
    qtbot.wait(20)

    qtbot.mouseClick(main_window.workflowTestPipelineBtn, Qt.LeftButton)
    qtbot.wait(20)

    assert main_window.workflowTestPipelineBtn.isChecked()
    assert not main_window.setupHost.isHidden()
    assert main_window.setup_panel.isHidden()
    assert not main_window.pipeline_step_test_panel.isHidden()

    main_window.actionFocusAnalysis.trigger()
    qtbot.wait(20)

    assert main_window.workflowTestPipelineBtn.isHidden()
    assert main_window.pipeline_step_test_panel.isHidden()
    assert not main_window.setup_panel.isHidden()


def test_pipeline_test_stage_list_excludes_acquisition(main_window):
    controller = main_window.pipeline_step_test_ctrl

    controller.refresh_stages()

    assert "acquisition" not in controller.panel._stage_names
    assert "preprocessing" in controller.panel._stage_names


def test_pipeline_test_stage_list_refreshes_on_pipeline_switch(
    main_window, monkeypatch
):
    controller = main_window.pipeline_step_test_ctrl

    class FakePipeline:
        def build_plan(self):
            return [
                ("acquisition", object()),
                ("preprocessing", object()),
                ("validation", object()),
            ]

    monkeypatch.setitem(controller.pipeline_map, "pendant", FakePipeline)
    main_window.setup_panel_ctrl._select_pipeline_button("pendant")
    controller.refresh_stages()

    assert controller.panel._stage_names == ["preprocessing", "validation"]


def test_pipeline_test_sandbox_does_not_mutate_live_settings_until_apply(main_window):
    controller = main_window.pipeline_step_test_ctrl
    live_settings = main_window.preprocessing_ctrl.settings
    original_value = live_settings.crop_to_roi

    controller._preprocessing_settings.crop_to_roi = not original_value
    controller._set_dirty(True)

    assert main_window.preprocessing_ctrl.settings.crop_to_roi is original_value

    controller.apply_sandbox()

    assert main_window.preprocessing_ctrl.settings.crop_to_roi is (not original_value)


def test_pipeline_test_stage_runs_with_prereqs_and_auto_payload(
    main_window, monkeypatch
):
    calls = {}

    class FakePipeline:
        def build_plan(self):
            return [("acquisition", object()), ("preprocessing", object())]

        def run_with_plan(self, *, only=None, include_prereqs=True, **kwargs):
            calls["only"] = only
            calls["include_prereqs"] = include_prereqs
            calls["kwargs"] = kwargs
            ctx = Context()
            ctx.timings_ms = {"preprocessing": 1.0}
            return ctx

    monkeypatch.setitem(main_window.pipeline_ctrl.pipeline_map, "sessile", FakePipeline)
    monkeypatch.setattr(
        main_window.pipeline_ctrl,
        "_auto_calibration_payload",
        lambda pipeline, image: (
            {
                "roi": (1, 2, 3, 4),
                "needle_rect": (5, 6, 7, 8),
                "drop_contour": object(),
            },
            [],
        ),
    )

    result = main_window.pipeline_ctrl.test_stage("preprocessing", {})

    assert result["ok"] is True
    assert calls["only"] == ["preprocessing"]
    assert calls["include_prereqs"] is True
    assert calls["kwargs"]["roi"] == (1, 2, 3, 4)
    assert calls["kwargs"]["needle_rect"] == (5, 6, 7, 8)


def test_workflow_bar_owns_analysis_and_source_controls(main_window):
    controller = main_window.setup_panel_ctrl

    assert main_window.workflowAnalysisLayout.indexOf(controller.sessileBtn) >= 0
    assert main_window.workflowAnalysisLayout.indexOf(controller.pendantBtn) >= 0
    assert main_window.workflowSourceLayout.indexOf(controller.singleModeRadio) == -1
    assert (
        main_window.workflowSourceLayout.indexOf(
            main_window.workflowSourceModeButtons[controller.MODE_SINGLE]
        )
        >= 0
    )
    assert (
        main_window.workflowSourceLayout.indexOf(
            main_window.workflowSourceModeButtons[controller.MODE_BATCH]
        )
        >= 0
    )
    assert (
        main_window.workflowSourceLayout.indexOf(
            main_window.workflowSourceModeButtons[controller.MODE_CAMERA]
        )
        >= 0
    )
    assert (
        main_window.workflowSourceLayout.indexOf(main_window.workflowSourceStackHost)
        >= 0
    )
    assert main_window.workflowSourceLayout.indexOf(controller.imagePathEdit) == -1
    assert main_window.workflowSourceLayout.indexOf(controller.batchPathEdit) == -1
    assert (
        main_window._workflow_source_page_layouts[controller.MODE_SINGLE].indexOf(
            main_window.workflowImageDisplay
        )
        >= 0
    )
    assert (
        main_window._workflow_source_page_layouts[controller.MODE_SINGLE].indexOf(
            controller.browseBtn
        )
        >= 0
    )
    assert controller.imagePathEdit.isHidden()
    assert controller.batchPathEdit.isHidden()
    assert controller.pipelineGroup is not None
    assert controller.pipelineGroup.isHidden()
    assert controller.sourceGroup is not None
    assert controller.sourceGroup.isHidden()


def test_workflow_source_mode_buttons_do_not_clip_when_narrow(main_window, qtbot):
    controller = main_window.setup_panel_ctrl
    main_window.resize(760, 700)
    main_window.show()
    qtbot.wait(80)

    host = main_window.workflowSourceHost
    widths = {
        button.width() for button in main_window.workflowSourceModeButtons.values()
    }
    heights = {
        button.height() for button in main_window.workflowSourceModeButtons.values()
    }
    assert len(widths) == 1
    assert len(heights) == 1
    for button in main_window.workflowSourceModeButtons.values():
        assert 40 <= button.width() <= 42
        assert button.height() == 32
        assert button.iconSize().width() == 18
        assert button.iconSize().height() == 18
        assert button.iconSize().width() < button.width() - 8
        assert button.iconSize().height() < button.height() - 8
        left = button.mapTo(host, button.rect().topLeft()).x()
        right = button.mapTo(host, button.rect().topRight()).x()
        assert left >= 0
        assert right <= host.width()

    assert main_window.workflowSourceModeButtons[controller.MODE_SINGLE].isChecked()
    qtbot.mouseClick(
        main_window.workflowSourceModeButtons[controller.MODE_BATCH], Qt.LeftButton
    )
    qtbot.wait(20)
    assert controller.current_mode() == controller.MODE_BATCH
    assert main_window.workflowSourceModeButtons[controller.MODE_BATCH].isChecked()
    assert not main_window.workflowSourceModeButtons[controller.MODE_SINGLE].isChecked()


def test_workflow_camera_page_uses_dropdown_and_settings_button(main_window, qtbot):
    _disconnect_camera_preview(main_window)
    controller = main_window.setup_panel_ctrl

    qtbot.mouseClick(
        main_window.workflowSourceModeButtons[controller.MODE_CAMERA], Qt.LeftButton
    )
    qtbot.wait(20)

    camera_layout = main_window._workflow_source_page_layouts[controller.MODE_CAMERA]
    assert camera_layout.indexOf(controller.sourceIdCombo) >= 0
    assert camera_layout.indexOf(main_window.workflowCameraSettingsBtn) >= 0
    assert camera_layout.indexOf(controller.framesSpin) == -1
    assert controller.framesSpin.isHidden()
    assert not controller.sourceIdCombo.isEditable()


def test_workflow_source_displays_compact_names_and_keep_full_paths(
    main_window, tmp_path
):
    controller = main_window.setup_panel_ctrl
    image_path = tmp_path / "drop.png"
    batch_dir = tmp_path / "batch images"
    batch_image = batch_dir / "frame001.png"
    image_path.write_bytes(b"not a real image")
    batch_dir.mkdir()
    batch_image.write_bytes(b"not a real image")

    controller.set_image_path(str(image_path))
    controller.set_batch_path(str(batch_dir))
    main_window._sync_workflow_source_displays()

    assert main_window.workflowImageDisplay.text() == "./drop.png"
    assert main_window.workflowImageDisplay.toolTip() == str(image_path)
    assert controller.image_path() == str(image_path)
    assert controller.imagePathEdit.text() == str(image_path)

    assert main_window.workflowBatchDisplay.text() == "./batch images"
    assert main_window.workflowBatchDisplay.toolTip() == str(batch_dir)
    assert controller.batch_path() == str(batch_dir)
    assert controller.batchPathEdit.text() == str(batch_dir)

    controller.batchModeRadio.click()
    assert controller.sourceIdCombo.itemText(0) == "frame001.png"
    assert controller.sourceIdCombo.itemData(0) == str(batch_image)
    assert controller.gather_run_params()["image"] == str(batch_image)


def test_camera_dropdown_uses_discovered_cameras(main_window, monkeypatch):
    from menipy.gui.controllers import setup_panel_controller as setup_module

    controller = main_window.setup_panel_ctrl
    _disconnect_camera_preview(main_window)
    monkeypatch.setattr(
        setup_module,
        "discover_available_cameras",
        lambda: [CameraDevice(0, "Built-in Camera"), CameraDevice(2, "USB Camera")],
    )

    controller.cameraModeRadio.click()

    assert not controller.sourceIdCombo.isEditable()
    assert controller.sourceIdCombo.itemText(0) == "Built-in Camera"
    assert controller.sourceIdCombo.itemData(0) == 0
    assert controller.sourceIdCombo.itemText(1) == "USB Camera"
    assert controller.sourceIdCombo.itemData(1) == 2


def test_camera_dropdown_falls_back_to_camera_zero(main_window, monkeypatch):
    from menipy.gui.controllers import setup_panel_controller as setup_module

    controller = main_window.setup_panel_ctrl
    _disconnect_camera_preview(main_window)
    monkeypatch.setattr(setup_module, "discover_available_cameras", lambda: [])

    controller.cameraModeRadio.click()

    assert not controller.sourceIdCombo.isEditable()
    assert controller.sourceIdCombo.count() == 1
    assert controller.sourceIdCombo.itemText(0) == "Camera 0"
    assert controller.sourceIdCombo.itemData(0) == 0
    assert controller.gather_run_params()["cam_id"] == 0


def test_camera_settings_button_applies_dialog_values(main_window, qtbot, monkeypatch):
    from menipy.gui.views import main_window as main_window_module

    _disconnect_camera_preview(main_window)
    controller = main_window.setup_panel_ctrl
    controller.cameraModeRadio.click()

    class FakeCameraSettingsDialog:
        Accepted = 1

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec(self):
            return self.Accepted

        def values(self):
            return {
                "device": 2,
                "frames": 12,
                "fps": 60,
                "width": 1280,
                "height": 720,
            }

    monkeypatch.setattr(
        main_window_module, "CameraSettingsDialog", FakeCameraSettingsDialog
    )

    qtbot.mouseClick(main_window.workflowCameraSettingsBtn, Qt.LeftButton)

    values = controller.camera_settings_values()
    assert values["device"] == 2
    assert values["frames"] == 12
    assert values["fps"] == 60
    assert values["width"] == 1280
    assert values["height"] == 720
    assert controller.gather_run_params()["frames"] == 12


def test_camera_preview_uses_selected_capture_config(main_window):
    controller = main_window.setup_panel_ctrl
    start = Mock()
    main_window.main_controller.camera_manager.camera_ctrl.start = start

    controller.apply_camera_settings(
        device=1,
        frames=5,
        fps=24,
        width=640,
        height=480,
    )
    controller.cameraModeRadio.click()

    start.assert_called()
    config = start.call_args.args[0]
    assert config.device == 1
    assert config.fps == 24
    assert config.width == 640
    assert config.height == 480


def _disconnect_camera_preview(main_window):
    try:
        main_window.setup_panel_ctrl.source_mode_changed.disconnect(
            main_window.main_controller.camera_manager.on_source_mode_changed
        )
    except (AttributeError, RuntimeError, TypeError):
        pass


def test_workflow_source_stack_stays_stable_when_maximized(main_window, qtbot):
    _disconnect_camera_preview(main_window)
    controller = main_window.setup_panel_ctrl

    main_window.resize(1500, 900)
    main_window.showMaximized()
    qtbot.wait(80)
    initial_height = main_window.workflowBar.height()

    for button, mode in (
        (controller.singleModeRadio, controller.MODE_SINGLE),
        (controller.batchModeRadio, controller.MODE_BATCH),
        (controller.cameraModeRadio, controller.MODE_CAMERA),
        (controller.singleModeRadio, controller.MODE_SINGLE),
    ):
        qtbot.mouseClick(button, Qt.LeftButton)
        qtbot.wait(80)
        assert main_window.workflowBar.height() == initial_height
        assert (
            main_window.workflowSourceStack.currentWidget()
            is main_window._workflow_source_pages[mode]
        )
        assert main_window.workflowBar.width() <= main_window.width()

    for button in (
        main_window.actionExportCsvBtn,
        main_window.toggleSetupBtn,
        main_window.toggleInspectBtn,
        main_window.toggleKeyResultsBtn,
    ):
        assert button.isVisible()
        right_edge = button.mapTo(main_window, button.rect().topRight()).x()
        assert right_edge <= main_window.width()


def test_preview_mapping_stays_inside_scene_after_source_toggles(main_window, qtbot):
    _disconnect_camera_preview(main_window)
    controller = main_window.setup_panel_ctrl
    image_view = main_window.preview_panel.image_view
    assert image_view is not None

    img = QImage(1400, 900, QImage.Format_RGB32)
    img.fill(Qt.black)
    image_view.set_image(img)
    image_view.fit_to_window()

    main_window.resize(1500, 900)
    main_window.showMaximized()
    qtbot.wait(80)
    for button in (
        controller.batchModeRadio,
        controller.cameraModeRadio,
        controller.singleModeRadio,
    ):
        qtbot.mouseClick(button, Qt.LeftButton)
        qtbot.wait(80)

    viewport_center = image_view.viewport().rect().center()
    scene_point = image_view.mapToScene(QPoint(viewport_center))
    assert image_view.scene().sceneRect().contains(scene_point)
    assert getattr(image_view, "_mode", None) == "fit"


def test_database_buttons_are_drawn_and_noop(main_window, qtbot):
    controller = main_window.setup_panel_ctrl
    for button in (
        controller.needleDbBtn,
        controller.dropDensityDbBtn,
        controller.fluidDensityDbBtn,
    ):
        assert button is not None
        assert button.isEnabled()
        assert not button.icon().isNull()
        qtbot.mouseClick(button, Qt.LeftButton)
        assert "Database selection is not connected yet" in (
            main_window.statusBar().currentMessage()
        )


def test_advanced_buttons_open_dialog_without_inline_expansion(main_window, qtbot):
    controller = main_window.setup_panel_ctrl
    dialog = main_window.advanced_workflow_dialog

    assert controller.advancedToggleBtn is not None
    assert not controller.advancedToggleBtn.isCheckable()
    assert not controller.sopGroup.isVisible()
    assert not controller.stepsGroup.isVisible()

    qtbot.mouseClick(controller.advancedToggleBtn, Qt.LeftButton)
    qtbot.wait(20)
    assert dialog.isVisible()
    assert controller.sopGroup.parent() is dialog
    assert controller.stepsGroup.parent() is dialog
    assert controller.sopGroup.isVisible()
    assert controller.stepsGroup.isVisible()

    dialog.close()
    qtbot.wait(20)
    assert not dialog.isVisible()
    assert not controller.sopGroup.isVisible()
    assert not controller.stepsGroup.isVisible()


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


def test_image_view_maximize_fit_is_deferred(main_window, qtbot):
    image_view = main_window.preview_panel.image_view
    assert image_view is not None
    img = QImage(1400, 900, QImage.Format_RGB32)
    img.fill(Qt.black)
    image_view.set_image(img)
    image_view.fit_to_window()

    main_window.show()
    qtbot.wait(20)
    main_window.showMaximized()
    qtbot.wait(100)

    assert not getattr(image_view, "_fit_resize_pending", True)
    assert getattr(image_view, "_mode", None) == "fit"


def test_setup_and_table_toggles_restore_splitter_sizes(main_window, qtbot):
    setup_toggle = main_window.toggleSetupBtn
    table_toggle = main_window.toggleInspectBtn

    assert setup_toggle.isChecked()
    assert table_toggle.isChecked()

    # Hide and restore setup rail.
    qtbot.mouseClick(setup_toggle, Qt.LeftButton)
    qtbot.wait(20)
    assert main_window.setupHost.isHidden()

    qtbot.mouseClick(setup_toggle, Qt.LeftButton)
    qtbot.wait(20)
    assert not main_window.setupHost.isHidden()
    assert main_window.rootSplitter.sizes()[0] > 0

    # Hide and restore table (inspect tabs).
    qtbot.mouseClick(table_toggle, Qt.LeftButton)
    qtbot.wait(20)
    assert main_window.inspectTabs.isHidden()

    qtbot.mouseClick(table_toggle, Qt.LeftButton)
    qtbot.wait(20)
    assert not main_window.inspectTabs.isHidden()
    assert main_window.workbenchSplitter.sizes()[1] > 0
