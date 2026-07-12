"""Blocking correctness checks for the versioned ADSA detector manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from adsa_conformance import evaluate_detection, generate_case  # noqa: E402

MANIFEST_PATH = Path(__file__).parent / "data" / "adsa_detector_manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_is_versioned_and_has_48_unique_cases():
    assert MANIFEST["schema_version"] == "1.0"
    assert MANIFEST["seed"] == 20260710
    assert len(MANIFEST["cases"]) == 48
    assert len({case["id"] for case in MANIFEST["cases"]}) == 48
    assert sum(case["expected_valid"] for case in MANIFEST["cases"]) == 32


@pytest.mark.parametrize("spec", MANIFEST["cases"], ids=lambda spec: spec["id"])
def test_detector_conformance_case(spec):
    generated = generate_case(spec, MANIFEST["seed"])
    result, metrics, reasons = evaluate_detection(generated, spec["pipeline"])
    accepted = not reasons

    if not spec["expected_valid"]:
        assert not accepted
        assert reasons
        return

    assert accepted, reasons
    assert result.drop_contour is not None
    assert result.needle_rect is not None
    thresholds = MANIFEST["thresholds"][spec["tier"]]
    assert metrics["iou"] >= thresholds["iou_min"]
    assert metrics["hausdorff_px"] <= thresholds["hausdorff_px_max"]
    assert metrics["needle_width_error_px"] <= thresholds["needle_px_max"]
    assert metrics["needle_center_error_px"] <= thresholds["needle_px_max"]
    assert metrics["apex_error_px"] <= thresholds["keypoint_px_max"]
    assert metrics["contact_error_px"] <= thresholds["keypoint_px_max"]
    if spec["pipeline"] == "sessile":
        assert metrics["baseline_y_error_px"] <= thresholds["baseline_y_px_max"]
        assert (
            metrics["baseline_angle_error_deg"]
            <= thresholds["baseline_angle_deg_max"]
        )
