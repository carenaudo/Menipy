"""Phase-D contracts, state classification, exports, GUI, and compatibility."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from menipy.common.sequence_acquisition import (
    SequenceAcquisitionError,
    load_image_sequence,
    load_video,
)
from menipy.common.temporal_sessile import (
    analyze_dynamic_sessile,
    export_dynamic_result,
)
from menipy.models.frame import Frame
from menipy.models.results import build_persisted_analysis
from menipy.models.temporal import SequenceMetadata
from menipy.pipelines.discover import PIPELINE_MAP


def _metadata(count: int = 30, fps: float = 10.0) -> SequenceMetadata:
    return SequenceMetadata(
        source_type="memory", source_id="test", sha256="0" * 64,
        width=200, height=160, fps=fps,
        timestamps_s=[index / fps for index in range(count)], frame_count=count,
    )


def _install_sequence_stubs(monkeypatch, *, gap: range = range(0)) -> None:
    detector_index = 0
    metric_index = 0

    def detect(_image, _pipeline):
        nonlocal detector_index
        index = detector_index
        detector_index += 1
        if index in gap:
            return {"detector_diagnostics": {"drop": {"accepted": False}}}
        width = 80.0 + (index if index < 10 else (10 if index < 20 else 30 - index))
        left, right = 100.0 - width / 2, 100.0 + width / 2
        contour = np.asarray([[left, 120.0], [70.0, 70.0], [130.0, 70.0], [right, 120.0]])
        return {
            "drop_contour": contour,
            "contact_points": ((left, 120.0), (right, 120.0)),
            "substrate_line": ((0.0, 120.0), (199.0, 120.0)),
            "needle_rect": (94, 0, 12, 40),
            "detector_diagnostics": {},
        }

    def metrics(*_args, **_kwargs):
        nonlocal metric_index
        index = metric_index
        metric_index += 1
        angle = 120.0 if index < 10 else (100.0 if index < 20 else 80.0)
        return {"theta_left_deg": angle, "theta_right_deg": angle, "method": "auto_residual"}

    monkeypatch.setattr("menipy.common.temporal_sessile.auto_detect_features", detect)
    monkeypatch.setattr("menipy.common.temporal_sessile.compute_sessile_metrics", metrics)


def test_manifest_has_48_versioned_cases():
    manifest = json.loads((Path(__file__).parent / "data" / "adsa_temporal_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 20260711
    assert manifest["case_count"] == len(manifest["cases"]) == 48
    assert sum(case["expected_valid"] for case in manifest["cases"]) == 32


def test_sequence_loader_natural_order_and_requires_fps(tmp_path):
    for name, value in (("frame10.png", 10), ("frame2.png", 2), ("frame1.png", 1)):
        assert cv2.imwrite(str(tmp_path / name), np.full((8, 10), value, dtype=np.uint8))
    with pytest.raises(SequenceAcquisitionError, match="sequence_fps_required"):
        load_image_sequence(tmp_path, fps=None)
    frames, metadata = load_image_sequence(tmp_path, fps=20.0)
    assert [int(frame.image[0, 0]) for frame in frames] == [1, 2, 10]
    assert metadata.timestamps_s == [0.0, 0.05, 0.1]


def test_video_loader_uses_fps_fallback_for_nonmonotonic_container_time(monkeypatch, tmp_path):
    source = tmp_path / "clip.avi"
    source.write_bytes(b"video-fixture-placeholder")

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0

        def isOpened(self):  # noqa: N802
            return True

        def read(self):
            if self.index >= 3:
                return False, None
            self.index += 1
            return True, np.full((6, 8, 3), self.index, dtype=np.uint8)

        def get(self, prop):
            return 25.0 if prop == cv2.CAP_PROP_FPS else 0.0

        def release(self):
            return None

    monkeypatch.setattr("menipy.common.sequence_acquisition.cv2.VideoCapture", FakeCapture)
    frames, metadata = load_video(source)
    assert len(frames) == 3
    assert metadata.timestamps_s == [0.0, 0.04, 0.08]


def test_dynamic_states_summary_and_raw_series_are_deterministic(monkeypatch):
    _install_sequence_stubs(monkeypatch)
    frames = [Frame(image=np.zeros((160, 200), dtype=np.uint8)) for _ in range(30)]
    first = analyze_dynamic_sessile(frames, _metadata(), px_per_mm=10.0, needle_diameter_mm=None)
    assert first.accepted
    assert any(frame.state == "advancing" for frame in first.frames)
    assert any(frame.state == "receding" for frame in first.frames)
    assert first.summary["theta_advancing_deg"] == pytest.approx(120.0)
    assert first.summary["theta_receding_deg"] == pytest.approx(80.0)
    assert first.summary["contact_angle_hysteresis_deg"] == pytest.approx(40.0)
    assert first.diagnostics["raw_values_promoted"] is True


def test_short_occlusion_reacquires_without_interpolation(monkeypatch):
    _install_sequence_stubs(monkeypatch, gap=range(12, 15))
    frames = [Frame(image=np.zeros((160, 200), dtype=np.uint8)) for _ in range(30)]
    result = analyze_dynamic_sessile(frames, _metadata(), px_per_mm=10.0, needle_diameter_mm=None)
    assert result.accepted
    assert all(not result.frames[index].accepted for index in range(12, 15))
    assert result.frames[15].segment_id == 1
    assert result.frames[15].diagnostics["reacquired_after_frames"] == 3


def test_missing_calibration_rejects_and_persistence_quarantines(monkeypatch):
    _install_sequence_stubs(monkeypatch)
    frames = [Frame(image=np.zeros((160, 200), dtype=np.uint8)) for _ in range(30)]
    result = analyze_dynamic_sessile(frames, _metadata(), px_per_mm=None, needle_diameter_mm=None)
    assert not result.accepted
    assert "dynamic_missing_calibration" in result.rejection_reasons
    context = type("ContextStub", (), {"results": {"theta_advancing_deg": 120.0}, "qa": {"ok": False, "rejection_reasons": result.rejection_reasons}})()
    assert build_persisted_analysis(context)["results"] == {}


def test_dynamic_exports_have_stable_long_form_without_contours(monkeypatch, tmp_path):
    _install_sequence_stubs(monkeypatch)
    frames = [Frame(image=np.zeros((160, 200), dtype=np.uint8)) for _ in range(30)]
    result = analyze_dynamic_sessile(frames, _metadata(), px_per_mm=10.0, needle_diameter_mm=None)
    paths = export_dynamic_result(result, tmp_path)
    assert list(paths) == ["json", "summary_csv", "frames_csv"]
    with paths["frames_csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert "contour" not in (reader.fieldnames or [])
        assert len(list(reader)) == 30


def test_pipeline_is_discovered_without_changing_static_sessile():
    assert "sessile_dynamic" in PIPELINE_MAP
    assert PIPELINE_MAP["sessile"].name == "sessile"


def test_cli_sequence_exports_three_canonical_files(tmp_path):
    from menipy.cli import main
    from tests.adsa_temporal_conformance import generate_temporal_case

    manifest = json.loads((Path(__file__).parent / "data" / "adsa_temporal_manifest.json").read_text(encoding="utf-8"))
    spec = dict(manifest["cases"][0])
    spec["frames"] = 20
    generated = generate_temporal_case(spec, manifest["seed"])
    source = tmp_path / "sequence"
    output = tmp_path / "output"
    source.mkdir()
    for index, image in enumerate(generated.images):
        assert cv2.imwrite(str(source / f"frame_{index:03d}.png"), image)
    code = main([
        "--pipeline", "sessile_dynamic", "--sequence-dir", str(source),
        "--fps", "30", "--px-per-mm", "20", "--out", str(output),
        "--db", str(tmp_path / "plugins.sqlite"),
        "--materials-db", str(tmp_path / "materials.sqlite"),
    ])
    assert code == 0
    assert {path.name for path in output.iterdir()} >= {"results.json", "results.csv", "results_frames.csv"}


def test_dynamic_timeline_navigation(qtbot):
    from menipy.gui.views.dynamic_timeline import DynamicTimelineWidget
    from menipy.models.temporal import TemporalFrameResult

    widget = DynamicTimelineWidget()
    qtbot.addWidget(widget)
    values = [TemporalFrameResult(frame_index=index, timestamp_s=index / 10, accepted=True, state="pinned", theta_left_deg=90, theta_right_deg=91) for index in range(4)]
    widget.set_sequence(values, 10.0)
    widget.slider.setValue(3)
    assert widget.label.text() == "4 / 4"
