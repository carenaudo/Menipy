"""Real Qt wiring, service identities, settings, and stale-result presentation."""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from menipy.gui.services.pipeline_runner import PipelineRunner, RunRequest
from menipy.gui.services.settings_service import AppSettings
from menipy.gui.services.sop_service import SopService
from menipy.gui.viewmodels.run_vm import RunViewModel
from menipy.models.results import MeasurementResult


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    from menipy.gui.services import settings_service
    from menipy.gui.views.main_window import MainWindow
    from menipy.models import results

    monkeypatch.setattr(
        settings_service, "_default_path", lambda: tmp_path / "settings.json"
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(results, "_results_history", None)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    widget = MainWindow()
    qtbot.addWidget(widget)
    widget.show()
    qtbot.wait(30)
    yield widget
    widget.runner.cancel()
    qtbot.waitUntil(lambda: not widget.runner.busy)
    widget.close()
    assert not errors


def test_required_services_and_shared_settings(window):
    assert isinstance(window.runner, PipelineRunner)
    assert isinstance(window.run_vm, RunViewModel)
    assert isinstance(window.sops, SopService)
    assert type(window.settings) is AppSettings
    assert window.results_panel_ctrl.settings is window.settings
    assert window.preview_panel.settings is window.settings
    window.settings.selected_pipeline = "pendant"
    window.results_panel_ctrl.settings.results_hidden_columns = {
        "sessile": ["volume_uL"]
    }
    window.results_panel_ctrl.settings.save()
    restored = AppSettings.load(window.settings.path)
    assert restored.selected_pipeline == "pendant"
    assert restored.results_hidden_columns == {"sessile": ["volume_uL"]}


def test_history_only_insert_preserves_selection_and_cards(window):
    panel = window.results_panel_ctrl
    old = MeasurementResult(
        id="old",
        timestamp=datetime.now(),
        pipeline="sessile",
        results={"diameter_mm": 2},
    )
    panel.add_measurement(old)
    cards = {key: label.text() for key, label in panel.metric_value_labels.items()}
    panel.add_measurement(
        MeasurementResult(
            id="late",
            timestamp=datetime.now(),
            pipeline="sessile",
            results={"diameter_mm": 99},
        ),
        activate=False,
    )
    selected = panel._measurement_for_row(panel.table.currentRow())
    assert selected.id == "old"
    assert cards == {
        key: label.text() for key, label in panel.metric_value_labels.items()
    }
    assert len(panel.history.measurements) == 2


def test_rejected_export_keeps_provenance_when_diagnostics_hidden(
    window, monkeypatch, tmp_path
):
    import csv
    import json

    from PySide6.QtWidgets import QFileDialog

    panel = window.results_panel_ctrl
    panel.settings.diagnostics_visible = False
    measurement = MeasurementResult(
        id="rejected-export",
        timestamp=datetime.now(),
        pipeline="sessile",
        file_path="original.png",
        accepted=False,
        rejection_reasons=["bad_fit"],
        run_metadata={"settings": {"needle_diameter_mm": 0.54}},
    )
    panel.add_measurement(measurement)
    destination = tmp_path / "export.csv"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args: (str(destination), "")
    )
    panel.export_csv()
    with destination.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert "Rejected" in rows[1] and "bad_fit" in rows[1]
    assert "original.png" in rows[1]
    assert any('"needle_diameter_mm":0.54' in cell for cell in rows[1])
    saved = json.loads(panel.history._history_file.read_text())
    assert saved["measurements"][0]["accepted"] is False
    assert saved["measurements"][0]["run_metadata"] == measurement.run_metadata


@pytest.mark.parametrize("kind", ["preprocessing", "edge"])
def test_preview_computation_runs_in_worker(window, qtbot, monkeypatch, kind):
    from PySide6.QtCore import QThread

    from menipy.gui.services import preview_execution

    gui_thread = QThread.currentThread()
    reported = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: reported.append(args))
    called = []

    def compute(ctx, settings):
        called.append(QThread.currentThread() != gui_thread)
        if kind == "preprocessing":
            raise ValueError("synthetic preview failure")
        return ctx

    target = (
        preview_execution.preprocessing
        if kind == "preprocessing"
        else preview_execution.edge_detection
    )
    monkeypatch.setattr(target, "run", compute)
    ctrl = (
        window.preprocessing_ctrl
        if kind == "preprocessing"
        else window.edge_detection_ctrl
    )
    ctrl.set_source(np.zeros((16, 16, 3), dtype=np.uint8))
    errors = []
    ctrl.errorOccurred.connect(errors.append)
    ctrl.run()
    assert window.runner.busy
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert called == [True]
    if kind == "preprocessing":
        assert errors == ["synthetic preview failure"]
        assert len(reported) == 1


def test_close_waits_responsively_and_suppresses_late_result(window, qtbot):
    entered, release = Event(), Event()
    finished = []

    def task(params, token):
        entered.set()
        release.wait(2)
        return "late"

    window.runner.finished.connect(finished.append)
    window.runner.submit(
        RunRequest.create("sessile", {}, operation="calibration"), task
    )
    qtbot.waitUntil(entered.is_set)
    window.close()
    assert window.isVisible()
    assert window.runner.busy
    assert "Stopping" in window.statusBar().currentMessage()
    release.set()
    qtbot.waitUntil(lambda: not window.isVisible())
    assert len(finished) == 1 and finished[0].state == "cancelled"


def test_pipeline_switch_keeps_run_controls_disabled_until_completion(window, qtbot):
    entered, release = Event(), Event()
    enabled = window.setup_panel_ctrl.runAllBtn.isEnabled()

    def task(params, token):
        entered.set()
        release.wait(2)

    window.runner.submit(
        RunRequest.create("sessile", {}, operation="calibration"), task
    )
    qtbot.waitUntil(entered.is_set)
    window.setup_panel_ctrl.pendantBtn.click()
    qtbot.wait(20)
    assert not window.setup_panel_ctrl.runAllBtn.isEnabled()
    assert window.setup_panel_ctrl.needleLengthSpin.isEnabled()
    release.set()
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert window.setup_panel_ctrl.runAllBtn.isEnabled() == enabled


@pytest.mark.parametrize("action", ["close", "edit"])
def test_calibration_discards_closed_or_changed_dialog(
    window, qtbot, monkeypatch, action
):
    from menipy.gui.dialogs.calibration_wizard_dialog import CalibrationWizardDialog
    from menipy.gui.services import calibration_service

    entered, release = Event(), Event()

    def task(params, token):
        entered.set()
        release.wait(2)
        return params["image"], object()

    monkeypatch.setattr(calibration_service, "calibration_task", task)
    dialog = CalibrationWizardDialog(
        np.zeros((16, 16, 3), dtype=np.uint8), parent=window
    )
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.run_detection()
    qtbot.waitUntil(entered.is_set)
    if action == "close":
        dialog.reject()
    else:
        dialog._manual_revision += 1
    release.set()
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert dialog.result is None
    dialog.close()


def test_settings_round_trip_across_real_startup_processes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
from pathlib import Path
sys.path.insert(0, 'tools')
from check_gui_execution import isolate
isolate(Path(sys.argv[1]))
from PySide6.QtWidgets import QApplication
from menipy.gui.views.main_window import MainWindow
from menipy.gui.services.settings_service import AppSettings
app = QApplication([])
w = MainWindow()
w.show()
app.processEvents()
if sys.argv[2] == 'write':
    w.resize(1380, 650)
    w.settings.selected_pipeline = 'pendant'
    w.settings.unit_system = 'CGS'
    w.settings.last_image_path = 'remembered.png'
    w.settings.overlay_config = {'line_width': 3}
    w.close()
else:
    assert w.runner is not None and w.sops is not None
    assert w.settings.selected_pipeline == 'pendant'
    assert w.settings.unit_system == 'CGS'
    assert w.settings.last_image_path == 'remembered.png'
    assert w.settings.overlay_config == {'line_width': 3}
    # Offscreen Qt may clamp width to the screen/minimum layout width.
    assert w.height() == 650, (w.width(), w.height())
    assert w.settings.main_window_geom_b64
    assert w.settings.guided_splitter_sizes
    w.close()
"""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
    for mode in ("write", "read"):
        result = subprocess.run(
            ["uv", "run", "--offline", "python", "-c", script, str(tmp_path), mode],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_required_service_initialization_failure_exits_with_message(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
from pathlib import Path
sys.path.insert(0, 'tools')
from check_gui_execution import isolate
isolate(Path(sys.argv[1]))
from PySide6.QtWidgets import QMessageBox
from menipy.gui.views import main_window
from menipy.gui import app
messages = []
QMessageBox.critical = lambda *args: messages.append(args)
def fail():
    raise RuntimeError('required service unavailable')
main_window.MainWindow = fail
assert app.main([]) == 1
assert len(messages) == 1
assert 'required service unavailable' in messages[0][2]
"""
    result = subprocess.run(
        ["uv", "run", "--offline", "python", "-c", script, str(tmp_path)],
        cwd=root,
        env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1"),
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
