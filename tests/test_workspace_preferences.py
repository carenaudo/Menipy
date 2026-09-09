"""Retention archival and persisted workspace preferences."""

import json
import subprocess
from pathlib import Path

import pytest

from menipy.gui.dialogs.settings_dialog import SettingsDialog
from menipy.gui.services.settings_service import AppSettings
from menipy.gui.services.workspace_preferences import WorkspacePreferences
from menipy.models.results import ResultsHistory
from tests import test_gui_execution_integration as integration
from tests.test_history_recovery_calibration import measurement

window = integration.window


def test_retention_archives_complete_records_before_removal(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    history = ResultsHistory(max_history=2)
    for index in range(3):
        history.add_measurement(measurement(str(index)))
    assert [m.id for m in history.measurements] == ["2", "1"]
    files = list(history.archive_directory.glob("*.json"))
    assert len(files) == 1
    archived = json.loads(files[0].read_text())["measurements"]
    assert archived[0]["id"] == "0"
    assert archived[0]["results"] == {"volume_uL": 2}
    history.clear_history()
    assert not history.measurements
    assert len(list(history.archive_directory.glob("*.json"))) == 2


def test_archive_failure_preserves_records_and_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    history = ResultsHistory(max_history=1)
    history.add_measurement(measurement("old"))
    with monkeypatch.context() as patch:
        patch.setattr(
            history,
            "_archive",
            lambda records: (_ for _ in ()).throw(OSError("disk full")),
        )
        history.add_measurement(measurement("new"))
        assert history.unsaved
        assert len(history.measurements) == 2
        history.clear_history()
        assert len(history.measurements) == 2
    assert history.retry_save()
    assert len(ResultsHistory().measurements) == 2


def test_preferences_apply_roundtrip_and_reset(window, qtbot, tmp_path, monkeypatch):
    dialog = SettingsDialog(window)
    qtbot.addWidget(dialog)
    original = window.settings
    dialog.units.setCurrentText("CGS")
    dialog.limit.setValue(250)
    dialog.checks["show_mode_labels"].setChecked(True)
    dialog.checks["diagnostics_visible"].setChecked(True)
    path = tmp_path / "preferences.json"
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName", lambda *args: (str(path), "")
    )
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getOpenFileName", lambda *args: (str(path), "")
    )
    dialog.export_file()
    dialog.load(WorkspacePreferences())
    dialog.import_file()
    assert dialog.limit.value() == 250
    assert window.settings.history_limit == 100
    dialog.apply()
    assert window.settings is original
    assert window.results_panel_ctrl.history.max_history == 250
    assert window.results_panel_ctrl.unit_system == "CGS"
    loaded = AppSettings.load(window.settings.path)
    assert loaded.history_limit == 250 and loaded.show_mode_labels
    assert loaded.diagnostics_visible
    subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "python",
            "-c",
            "from pathlib import Path; from menipy.gui.services.settings_service import AppSettings; "
            "import sys; s=AppSettings.load(Path(sys.argv[1])); "
            "assert s.history_limit==250 and s.unit_system=='CGS' and s.show_mode_labels and s.diagnostics_visible",
            str(window.settings.path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dialog.load(WorkspacePreferences())
    assert window.settings.history_limit == 250  # Reset is a draft until Apply.
    dialog.apply()
    assert window.settings.history_limit == 100
    assert window.settings.unit_system == "SI"


def test_import_rejects_unsupported_and_settings_failure_is_transactional(
    tmp_path, monkeypatch
):
    settings = AppSettings(path=tmp_path / "settings.json")
    settings.save()
    old = settings.path.read_bytes()
    with pytest.raises(ValueError):
        WorkspacePreferences.model_validate({"gpu_enabled": True})
    with pytest.raises(ValueError):
        WorkspacePreferences(history_limit=0)
    monkeypatch.setattr(
        "menipy.gui.services.settings_service.os.replace",
        lambda *args: (_ for _ in ()).throw(PermissionError("blocked")),
    )
    with pytest.raises(PermissionError):
        WorkspacePreferences(unit_system="CGS").apply(settings)
    assert settings.unit_system == "SI"
    assert settings.path.read_bytes() == old
