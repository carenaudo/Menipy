"""Persistence fault injection and calibration publication contracts."""

import errno
import json
from datetime import datetime
from pathlib import Path

import pytest

from menipy.gui.services.calibration_provenance import describe_calibration
from menipy.models.calibration import CalibrationProvenance
from menipy.models.context import Context
from menipy.models.results import (
    MeasurementResult,
    ResultsHistory,
    build_persisted_analysis,
)
from tests import test_gui_execution_integration as integration

window = integration.window


def measurement(identifier):
    return MeasurementResult(
        id=identifier,
        timestamp=datetime.now(),
        pipeline="sessile",
        results={"volume_uL": 2},
    )


@pytest.mark.parametrize("failure", ["permission", "disk_full", "interrupt"])
def test_atomic_failure_keeps_last_good_file_and_recovery(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    history = ResultsHistory(max_history=2)
    history.add_measurement(measurement("old"))
    original = history._history_file.read_bytes()
    with monkeypatch.context() as patch:

        def fail(*args):
            if failure == "permission":
                raise PermissionError("Read-only destination")
            raise OSError(
                errno.ENOSPC if failure == "disk_full" else errno.EIO, failure
            )

        patch.setattr(
            "menipy.models.results.os.replace"
            if failure == "permission"
            else "menipy.models.results.os.fsync",
            fail,
        )
        history.add_measurement(measurement("new"))
        assert history.unsaved and history.save_error
        assert history._history_file.read_bytes() == original
        history.add_measurement(measurement("newest"))
        assert len(history.measurements) == 3
        assert not list(history._data_dir.glob("*.tmp"))
    recovery = tmp_path / "recovery.json"
    history.export_recovery(recovery)
    assert len(json.loads(recovery.read_text())["measurements"]) == 3
    assert history.unsaved
    assert history.retry_save()
    assert not history.unsaved and history.save_error is None
    assert len(ResultsHistory().measurements) == 3


def test_history_failure_banner_and_retry(window, monkeypatch):
    panel = window.results_panel_ctrl
    with monkeypatch.context() as patch:
        patch.setattr(
            panel.history,
            "_write_atomic",
            lambda path: (_ for _ in ()).throw(PermissionError("blocked")),
        )
        panel.add_measurement(measurement("unsaved"))
        assert not panel.persistence_notice.isHidden()
        assert "blocked" in panel.persistence_label.text()
    panel.history.retry_save()
    assert panel.persistence_notice.isHidden()


@pytest.mark.parametrize(
    "origin,enabled",
    [("manual", True), ("measured", True), ("estimated", False), ("missing", False)],
)
def test_calibration_publication_preserves_scientific_qa(origin, enabled):
    ctx = Context(
        results={"volume_uL": 2},
        qa={"ok": True},
        calibration_provenance=CalibrationProvenance(origin=origin, px_per_mm=20),
    )
    persisted = build_persisted_analysis(ctx)
    assert persisted["accepted"]
    assert bool(persisted["results"]) is enabled
    assert (
        persisted["diagnostics"]["calibration"]["physical_values_withheld"]
        is not enabled
    )
    assert ctx.results == {"volume_uL": 2}


def test_component_warning_survives_high_overall_confidence():
    provenance = describe_calibration(
        {
            "scale": {"px_per_mm": 10},
            "calibration_provenance": {
                "origin": "manual",
                "component_confidence": {"overall": 0.81, "substrate": 0.25},
            },
        },
        Context(),
        [],
    )
    assert provenance.physical_values_enabled
    assert any("substrate" in warning for warning in provenance.warnings)


def test_uncalibrated_result_is_visible_in_history_and_export(window, tmp_path):
    ctx = Context(
        results={"volume_uL": 99},
        calibration_provenance=CalibrationProvenance(
            origin="estimated", px_per_mm=10, warnings=["Review calibration"]
        ),
    )
    result = MeasurementResult(
        id="estimated",
        timestamp=datetime.now(),
        pipeline="sessile",
        **build_persisted_analysis(ctx),
    )
    panel = window.results_panel_ctrl
    panel.add_measurement(result)
    assert result.display_status == "Uncalibrated"
    assert "estimated" in panel.calibration_notice.text()
    path = tmp_path / "export.csv"
    assert panel.history.export_csv(path)
    assert "estimated" in path.read_text() and "Uncalibrated" in path.read_text()


def test_stage_detection_preserves_manual_calibration(monkeypatch):
    import numpy as np

    from menipy.common.auto_calibrator import AutoCalibrator, CalibrationResult
    from menipy.gui.services.calibration_service import prepare_stage_calibration

    monkeypatch.setattr(
        AutoCalibrator,
        "detect_all",
        lambda self: CalibrationResult(
            needle_rect=(1, 1, 40, 90), roi_rect=(0, 0, 100, 100)
        ),
    )
    parameters, _ = prepare_stage_calibration(
        "sessile",
        {
            "image": np.zeros((120, 120, 3), dtype=np.uint8),
            "scale": {"px_per_mm": 12},
            "roi": (2, 2, 80, 80),
            "calibration_provenance": {"origin": "manual"},
        },
    )
    assert parameters["scale"] == {"px_per_mm": 12}
    assert parameters["roi"] == (2, 2, 80, 80)
    assert parameters["calibration_provenance"]["origin"] == "manual"
