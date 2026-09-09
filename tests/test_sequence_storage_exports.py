"""Bounded acquisition, cancellation cleanup and canonical history exports."""

import csv
import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from menipy.common.cancellation import AnalysisCancelled
from menipy.common.sequence_acquisition import load_image_sequence, load_video
from menipy.models.frame_store import DiskFrameStore
from menipy.models.results import MeasurementResult, ResultsHistory
from tests import test_gui_execution_integration as integration

window = integration.window


def test_export_actions_keep_view_and_history_scope_separate(
    window, monkeypatch, tmp_path
):
    panel = window.results_panel_ctrl
    panel.history.measurements = [
        MeasurementResult(
            id=mode,
            timestamp=datetime.now(timezone.utc),
            pipeline=mode,
            file_path=f"{mode}.png",
            results={"diameter_mm": 1.23456789},
        )
        for mode in ("sessile", "pendant")
    ]
    panel.set_pipeline_filter("sessile")
    panel.update_history()
    for index, key in enumerate(panel._raw_headers):
        if key == "diameter_mm":
            panel.table.setColumnHidden(index, True)
    assert "all history" in window.actionExportCsvBtn.text()
    assert "current view" in panel.export_button.text()
    path = tmp_path / "history.csv"
    monkeypatch.setattr(
        "PySide6.QtWidgets.QFileDialog.getSaveFileName", lambda *args: (str(path), "")
    )
    window.main_controller.export_results_csv()
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 2
    assert records[0]["diameter_mm"] == "1.23456789"
    assert records[1]["file_path"] == "pendant.png"
    panel.export_csv()
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 1
    assert not any("diameter" in key.lower() for key in records[0])


@pytest.mark.parametrize("scale,gap", [(10.0, range(12, 15)), (None, range(12, 19))])
def test_temporal_disk_output_matches_eager_including_initial_scale(
    monkeypatch, scale, gap
):
    from menipy.common.temporal_sessile import analyze_dynamic_sessile
    from menipy.models.frame import Frame
    from tests.test_phase_d_dynamic_sessile import _install_sequence_stubs, _metadata

    frames = [Frame(image=np.zeros((160, 200), np.uint8)) for _ in range(30)]
    store = DiskFrameStore()
    for frame in frames:
        store.append(frame.image)
    store.timestamps_s = _metadata().timestamps_s
    try:
        _install_sequence_stubs(monkeypatch, gap=gap)
        eager = analyze_dynamic_sessile(
            frames, _metadata(), px_per_mm=scale, needle_diameter_mm=1.2
        )
        _install_sequence_stubs(monkeypatch, gap=gap)
        disk = analyze_dynamic_sessile(
            store, _metadata(), px_per_mm=scale, needle_diameter_mm=1.2
        )
        assert disk.model_dump(mode="json") == eager.model_dump(mode="json")
    finally:
        store.close()


@pytest.mark.parametrize("planned", [False, True])
def test_pipeline_cancellation_between_stages_closes_store(planned):
    from menipy.pipelines.base import PipelineBase

    store = DiskFrameStore()

    class CancelPipeline(PipelineBase):
        def do_acquisition(self, ctx):
            ctx.sequence_store = store
            return ctx

        def do_preprocessing(self, ctx):
            raise AnalysisCancelled()

    pipeline = CancelPipeline()
    with pytest.raises(AnalysisCancelled):
        if planned:
            pipeline.run_with_plan(only=["acquisition", "preprocessing"])
        else:
            pipeline.run()
    assert store.closed


def test_disk_sequence_matches_eager_and_is_owned(tmp_path):
    for index in (10, 1, 2):
        cv2.imwrite(
            str(tmp_path / f"frame{index}.png"), np.full((8, 10, 3), index, np.uint8)
        )
    eager, metadata = load_image_sequence(tmp_path, fps=20)
    stored, stored_metadata = load_image_sequence(tmp_path, fps=20, disk_backed=True)
    try:
        assert stored_metadata == metadata
        assert len(stored) == len(eager)
        for left, right in zip(stored, eager):
            np.testing.assert_array_equal(left.image, right.image)
            assert left.ms_from_start == right.ms_from_start
        stored[0].image[:] = 99
        assert stored[0].image[0, 0, 0] == 1
        assert not any(isinstance(v, np.ndarray) for v in vars(stored).values())
    finally:
        stored.close()
    assert stored.closed


def test_video_disk_matches_eager_and_releases_on_cancel(monkeypatch, tmp_path):
    source = tmp_path / "clip.avi"
    source.write_bytes(b"fixture")
    released = []

    class Capture:
        def __init__(self, _):
            self.index = 0

        def isOpened(self):
            return True

        def read(self):
            self.index += 1
            return (
                (True, np.full((8, 10, 3), self.index, np.uint8))
                if self.index <= 30
                else (False, None)
            )

        def get(self, prop):
            return 20 if prop == cv2.CAP_PROP_FPS else 0

        def release(self):
            released.append(True)

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    eager, metadata = load_video(source)
    stored, disk_metadata = load_video(source, disk_backed=True)
    assert disk_metadata == metadata
    for left, right in zip(eager, stored):
        np.testing.assert_array_equal(left.image, right.image)
    stored.close()
    stores = []
    original = DiskFrameStore.close

    def close(self):
        stores.append(self)
        original(self)

    monkeypatch.setattr(DiskFrameStore, "close", close)

    checkpoints = 0

    def cancel():
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 3:
            raise AnalysisCancelled()

    with pytest.raises(AnalysisCancelled):
        load_video(source, disk_backed=True, check_cancelled=cancel)
    assert len(released) == 3
    assert stores[0].closed
    assert len(stores[0]) == 2


def test_runtime_provenance_hashes_registered_sources(monkeypatch):
    import hashlib
    from pathlib import Path

    from menipy.common import registry
    from menipy.common.runtime_provenance import runtime_provenance

    registered = registry.Registry("test")
    registered.register("example", test_runtime_provenance_hashes_registered_sources)
    registered.register("native", int)
    monkeypatch.setattr(registry, "TEST_PROVENANCE", registered, raising=False)
    result = runtime_provenance()
    source = Path(__file__)
    assert (
        result["registered_source_sha256"][str(source)]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )


def test_machine_export_full_dates_precision_and_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    history = ResultsHistory()
    accepted = MeasurementResult(
        id="job",
        timestamp=datetime(2026, 9, 8, 12, tzinfo=timezone.utc),
        pipeline="sessile",
        file_path="original.png",
        results={"theta_left_deg": 91.123456789, "schema_version": "1.0"},
        diagnostics={"calibration": {"origin": "manual", "px_per_mm": 42}},
        run_metadata={
            "settings": {"threshold": 42},
            "runtime": {"registered_source_sha256": {"plugin.py": "abc"}},
        },
    )
    rejected = accepted.model_copy(
        update={
            "id": "rejected",
            "accepted": False,
            "results": {},
            "rejection_reasons": ["reason;one", "reason two"],
        }
    )
    history.measurements = [accepted, rejected]
    assert history.export_csv(tmp_path / "machine.csv")
    with (tmp_path / "machine.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert datetime.fromisoformat(rows[0]["timestamp"]) == accepted.timestamp
    assert float(rows[0]["theta_left_deg"]) == accepted.results["theta_left_deg"]
    assert rows[0]["file_path"] == "original.png"
    assert rows[0]["px_per_mm"] == "42"
    assert rows[0]["export_schema_version"] == "1.0"
    assert json.loads(rows[0]["run_metadata_json"]) == accepted.run_metadata
    assert json.loads(rows[1]["results_json"]) == {}
    assert json.loads(rows[1]["rejection_reasons_json"]) == rejected.rejection_reasons
