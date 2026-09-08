"""A08–A10 behavioral coverage using isolated real windows."""

import os
import subprocess

import numpy as np
import pytest

from menipy.gui.dialogs.calibration_wizard_dialog import CalibrationWizardDialog
from menipy.gui.services.sop_service import SopService
from menipy.models.preset import AnalysisPreset
from tests import test_gui_execution_integration as integration

window = integration.window


def test_preset_roundtrip_restores_full_settings(window, tmp_path):
    window.setup_panel_ctrl.needleLengthSpin.setValue(2.5)
    window.preprocessing_ctrl.settings.resize.target_width = 777
    window.edge_detection_ctrl.settings.canny_threshold1 = 42
    window.preprocessing_ctrl.update_markers({"drop_center": [15, 25]})
    window.settings.pipeline_settings = {"contact_angle_method": "tangent"}
    preset = window.preset_ctrl.capture("Repeatable")
    window.preset_ctrl.save(preset)
    restored_sop = SopService().get("sessile", "Repeatable")
    restored = AnalysisPreset.model_validate(restored_sop.params["__preset__"])
    path = tmp_path / "preset.json"
    path.write_text(restored.model_dump_json(), encoding="utf-8")
    restored = AnalysisPreset.model_validate_json(path.read_text(encoding="utf-8"))
    window.setup_panel_ctrl.pendantBtn.click()
    window.setup_panel_ctrl.needleLengthSpin.setValue(1)
    window.edge_detection_ctrl.settings.canny_threshold1 = 99
    window.settings.pipeline_settings = {"unrelated": True}
    assert window.preset_ctrl.apply(restored, confirm=False)
    assert window.setup_panel_ctrl.current_pipeline_name() == "sessile"
    assert window.setup_panel_ctrl.needleLengthSpin.value() == 2.5
    assert window.preprocessing_ctrl.settings.resize.target_width == 777
    assert window.edge_detection_ctrl.settings.canny_threshold1 == 42
    assert window.preprocessing_ctrl.markers.drop_center == (15, 25)
    assert window.settings.pipeline_settings == {"contact_angle_method": "tangent"}


def test_conflicting_preset_is_rejected_before_mutation(window):
    preset = window.preset_ctrl.capture("Missing plugin")
    preset.plugins["not_installed.py"] = "different"
    before = window.setup_panel_ctrl.needleLengthSpin.value()
    with pytest.raises(ValueError, match="Missing"):
        window.preset_ctrl.apply(preset, confirm=False)
    assert window.setup_panel_ctrl.needleLengthSpin.value() == before


def test_preset_reads_in_fresh_process(window, tmp_path):
    window.setup_panel_ctrl.gravitySpin.setValue(9.7)
    window.preprocessing_ctrl.settings.resize.target_width = 888
    window.preset_ctrl.save(window.preset_ctrl.capture("Restart"))
    script = """
from pathlib import Path
import sys
from menipy.gui.services.sop_service import SopService
from menipy.models.preset import AnalysisPreset
Path.home = classmethod(lambda cls: Path(sys.argv[1]))
sop = SopService().get('sessile', 'Restart')
preset = AnalysisPreset.model_validate(sop.params['__preset__'])
assert preset.controls['gravitySpin'] == 9.7
assert preset.preprocessing.resize.target_width == 888
"""
    result = subprocess.run(
        ["uv", "run", "--offline", "python", "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, PYTHONUTF8="1"),
    )
    assert result.returncode == 0, result.stderr


def test_source_readiness_and_calibration_invalidation(window, tmp_path, qtbot):
    assert not window.setup_panel_ctrl.runAllBtn.isEnabled()
    assert "Select" in window.readiness_ctrl.notice.text()
    path = tmp_path / "image.png"
    path.write_bytes(b"source identity")
    window.setup_panel_ctrl.set_image_path(str(path))
    window._last_calibration_result = object()
    window.setup_panel_ctrl.batchModeRadio.click()
    empty = tmp_path / "empty"
    empty.mkdir()
    window.setup_panel_ctrl.set_batch_path(str(empty))
    qtbot.wait(40)
    assert window._last_calibration_result is None
    assert not window.setup_panel_ctrl.runAllBtn.isEnabled()
    assert "Previous results" in window.readiness_ctrl.previous_notice.text()
    for button in window.workflowSourceModeButtons.values():
        assert button.accessibleName()


@pytest.mark.parametrize("size", [(1200, 800), (1366, 768)])
def test_calibration_initial_fit_and_status_readable(qtbot, size):
    image = np.zeros((1800, 1200, 3), dtype=np.uint8)
    dialog = CalibrationWizardDialog(image)
    qtbot.addWidget(dialog)
    dialog.resize(*size)
    dialog.show()
    qtbot.wait(40)
    pixmap = dialog._preview_label.pixmap()
    viewport = dialog._preview_scroll.viewport().size()
    assert pixmap.height() <= viewport.height()
    assert pixmap.width() <= viewport.width()
    assert abs(pixmap.width() / pixmap.height() - 2 / 3) < 0.02
    for region in dialog._region_widgets.values():
        region.status.setText("Doubtful (25%)")
        assert region.status.width() >= region.status.fontMetrics().horizontalAdvance(
            "Doubtful (25%)"
        )
    dialog.close()


def test_calibration_file_preview_loads_without_detection(qtbot, tmp_path):
    import cv2

    path = tmp_path / "preview.png"
    cv2.imwrite(str(path), np.full((600, 400, 3), 150, dtype=np.uint8))
    dialog = CalibrationWizardDialog(str(path))
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.original_image.shape == (600, 400, 3))
    assert dialog.result is None
    assert dialog._detect_btn.isEnabled()
    dialog.close()
