"""Folder lifecycle tests independent of scientific methodology changes."""

import csv
import time

from PySide6.QtCore import Qt, QThread, QTimer

from menipy.gui.services.pipeline_runner import RunCompletion
from menipy.models.context import Context
from tests import test_gui_execution_integration as integration

window = integration.window


def prepare(window, tmp_path):
    for name in ("a.png", "b.jpg", "c.tif"):
        (tmp_path / name).write_bytes(b"test input")
    window.setup_panel_ctrl.batchModeRadio.click()
    window.setup_panel_ctrl.set_batch_path(str(tmp_path))
    return window.folder_ctrl


def test_dynamic_button_accessible_and_sequence_action(window, qtbot):
    button = window.setup_panel_ctrl.dynamicSessileBtn
    assert button.isVisible()
    button.setFocus()
    qtbot.keyClick(button, Qt.Key_Space)
    assert window.setup_panel_ctrl.current_pipeline_name() == "sessile_dynamic"
    assert window.setup_panel_ctrl.dynamicFpsSpin.isVisible()
    assert window.setup_panel_ctrl.runAllBtn.text() == "Run sequence"
    window.setup_panel_ctrl.batchModeRadio.click()
    assert window.folder_ctrl.button.isHidden()
    assert not window.workflowSourceModeButtons["camera"].isEnabled()


def test_folder_failure_continues_export_and_retry(
    window, qtbot, tmp_path, monkeypatch
):
    panel = prepare(window, tmp_path)
    calls = []

    def execute(request, token):
        assert QThread.currentThread() != window.thread()
        calls.append(request)
        if request.source.endswith("b.jpg") and len(calls) < 4:
            raise ValueError("Corrupt image")
        return RunCompletion(
            request, "completed", ctx=Context(results={"volume_uL": 2.0})
        )

    monkeypatch.setattr("menipy.gui.services.folder_execution.execute_request", execute)
    panel.start()
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert list(panel.outcomes.values()) == ["accepted", "failed", "accepted"]
    assert len(panel.records) == 2
    assert len({r.job_id for r in calls}) == 3
    path = tmp_path / "batch.csv"
    panel.write_csv(path)
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert rows[1]["details"] == "Corrupt image"
    assert rows[0]["volume_uL"] == "2.0"
    panel.retry_failed()
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert len(calls) == 4
    assert len(panel.records) == 3
    assert calls[-1].job_id != calls[1].job_id
    panel.retry_failed()
    assert not window.runner.busy


def test_folder_cancel_preserves_completed_files_and_heartbeat(
    window, qtbot, tmp_path, monkeypatch
):
    panel = prepare(window, tmp_path)
    beats = []
    timer = QTimer(window)
    timer.setInterval(20)
    timer.timeout.connect(lambda: beats.append(1))
    timer.start()

    def execute(request, token):
        if request.source.endswith("b.jpg"):
            while True:
                time.sleep(0.01)
                token.check()
        return RunCompletion(
            request, "completed", ctx=Context(results={"volume_uL": 2.0})
        )

    monkeypatch.setattr("menipy.gui.services.folder_execution.execute_request", execute)
    panel.start()
    qtbot.waitUntil(lambda: len(panel.records) == 1 and len(beats) >= 5)
    panel.stop.click()
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert list(panel.outcomes.values()) == ["accepted", "cancelled", "cancelled"]
    assert len(panel.records) == 1
    timer.stop()


def test_folder_rejection_and_setup_changes_preserve_identity(
    window, qtbot, tmp_path, monkeypatch
):
    panel = prepare(window, tmp_path)
    requests = []

    def execute(request, token):
        requests.append(request)
        time.sleep(0.05)
        return RunCompletion(
            request,
            "completed",
            ctx=Context(
                results={},
                qa={"ok": False, "rejection_reasons": ["synthetic_rejection"]},
            ),
        )

    monkeypatch.setattr("menipy.gui.services.folder_execution.execute_request", execute)
    panel.start()
    window.setup_panel_ctrl.pendantBtn.click()
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert len(panel.records) == 3
    assert all(
        r["pipeline"] == "sessile" and r["results"] == {}
        for r in panel.records.values()
    )
    assert all(s == "rejected" for s in panel.outcomes.values())
    assert all(r.parameters["auto_calibrate"] for r in requests)


def test_dynamic_folder_submits_one_sequence(window, tmp_path, monkeypatch):
    prepare(window, tmp_path)
    window.setup_panel_ctrl.dynamicSessileBtn.click()
    window.setup_panel_ctrl.dynamicFpsSpin.setValue(30)
    submitted = []
    monkeypatch.setattr(
        window.run_vm, "submit", lambda request: submitted.append(request)
    )
    window.setup_panel_ctrl.runAllBtn.click()
    assert len(submitted) == 1
    assert submitted[0].parameters["sequence_path"] == str(tmp_path)
    assert submitted[0].parameters["sequence_fps"] == 30
