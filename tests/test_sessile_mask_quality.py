"""Regression guards for sessile drop segmentation quality.

These pin the defects behind the post-calibration contour notch: a holey
adaptive-threshold mask winning candidate selection on raw area, and an Otsu
fallback that cut away the contact band.
"""

from __future__ import annotations

import json
from datetime import datetime
from importlib import import_module
from typing import get_args

import cv2
import numpy as np
import pytest

from menipy.common.auto_calibrator import AutoCalibrator
from menipy.common.sessile_detection import (
    contour_fill_ratio,
    contour_solidity,
    detect_sessile_drop_contour,
)
from menipy.models.config import EdgeDetectionSettings
from menipy.models.results import MeasurementResult, to_builtin

SAMPLE = "data/samples/sessile_3.jpeg"
SUBSTRATE_Y = 249


@pytest.fixture(scope="module")
def sample_image() -> np.ndarray:
    image = cv2.imread(SAMPLE)
    assert image is not None, f"missing sample {SAMPLE}"
    return image


def test_detected_contour_is_not_notched(sample_image: np.ndarray) -> None:
    """A specular reflection must not carve a notch into the drop boundary.

    The adaptive-threshold mask leaves the drop interior full of holes; where
    one breaks through to the boundary the external contour dives inward. That
    candidate scored 0.898 solidity and won on raw area alone.
    """
    detection = detect_sessile_drop_contour(sample_image, substrate_y=SUBSTRATE_Y)

    assert detection.contour is not None
    assert detection.solidity is not None and detection.solidity >= 0.95
    assert detection.fill_ratio is not None and detection.fill_ratio >= 0.90


def test_detected_contour_keeps_the_contact_band(sample_image: np.ndarray) -> None:
    """The silhouette must reach the baseline, not stop short of it.

    The Otsu fallback used to mask from ``substrate_y - 4``, deleting the
    contact region the tangent fit needs and collapsing the drop from 46 px to
    39 px tall.
    """
    detection = detect_sessile_drop_contour(sample_image, substrate_y=SUBSTRATE_Y)
    contour = np.asarray(detection.contour, dtype=float)

    assert contour[:, 1].max() >= SUBSTRATE_Y - 1
    assert (contour[:, 1].max() - contour[:, 1].min()) >= 44


def test_calibration_contour_matches_detection(sample_image: np.ndarray) -> None:
    """The contour calibration hands to a run carries the same quality."""
    result = AutoCalibrator(sample_image, "sessile").detect_all()

    assert result.drop_contour is not None
    contour = np.asarray(result.drop_contour).reshape(-1, 1, 2).astype(np.int32)
    assert contour_solidity(contour) >= 0.95

    left, right = result.contact_points
    assert right[0] - left[0] > 150


EDGE_METHODS = sorted(
    set(get_args(EdgeDetectionSettings.model_fields["method"].annotation))
)

# Detectors that live in their own plugin module rather than edge_detectors.py.
STANDALONE_DETECTOR_MODULES = {
    "subpixel": "auto_subpixel_edge",
    "auto_adaptive": "auto_adaptive_edge",
}


@pytest.mark.parametrize("method", EDGE_METHODS)
def test_every_offered_edge_detector_is_registered(method: str) -> None:
    """Every method the settings offer must resolve to a real detector.

    An unregistered name silently resolves to the built-in fallback, which
    ignores the detector's parameters entirely -- selecting Canny in the GUI
    used to do nothing.
    """
    # Importing detection_helpers first is what puts plugins/ on sys.path, so
    # the plugin modules below are loaded through importlib rather than import
    # statements, which a formatter would be free to hoist above it.
    import menipy.common.detection_helpers  # noqa: F401
    from menipy.common.registry import EDGE_DETECTORS

    import_module("edge_detectors")
    standalone = STANDALONE_DETECTOR_MODULES.get(method)
    if standalone is not None:
        pytest.importorskip(standalone)
        import_module(standalone)

    assert method in EDGE_DETECTORS, f"edge detector {method!r} is not registered"


def test_measurement_history_survives_numpy_scalars() -> None:
    """Numpy scalars in free-form results must not break history saving.

    ``tuple(arr.astype(int))`` yields ``np.int64``, and pydantic raises on it
    before ``json.dumps`` is reached, so the whole history fails to save.
    """
    record = MeasurementResult(
        id="test",
        timestamp=datetime(2026, 1, 1),
        pipeline="sessile",
        results={
            "contact_line": (
                (np.int64(217), np.int64(249)),
                (np.int64(399), np.int64(249)),
            ),
            "theta_left_deg": np.float64(49.9),
            "profile": np.array([[1, 2], [3, 4]]),
        },
    )

    payload = json.dumps({"measurements": [record.model_dump(mode="json")]})
    reloaded = json.loads(payload)["measurements"][0]

    assert reloaded["results"]["contact_line"] == [[217, 249], [399, 249]]
    assert reloaded["results"]["theta_left_deg"] == pytest.approx(49.9)


def test_solidity_separates_a_notched_boundary_from_a_clean_one() -> None:
    """The gate's discriminator must actually discriminate."""
    clean = np.zeros((120, 200), np.uint8)
    cv2.ellipse(clean, (100, 100), (80, 50), 0, 180, 360, 255, -1)

    notched = clean.copy()
    # A fjord cut in from the boundary, as a broken-through interior hole makes.
    cv2.rectangle(notched, (128, 58), (146, 84), 0, -1)

    def outer(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return max(contours, key=cv2.contourArea)

    assert contour_solidity(outer(clean)) >= 0.97
    assert contour_solidity(outer(notched)) < 0.95


def test_fill_ratio_detects_a_holey_interior() -> None:
    """Interior holes are invisible to contourArea but must not be to us."""
    solid = np.zeros((120, 200), np.uint8)
    cv2.ellipse(solid, (100, 100), (80, 50), 0, 180, 360, 255, -1)

    holey = solid.copy()
    for cx in range(50, 155, 24):
        cv2.circle(holey, (cx, 80), 8, 0, -1)

    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # Same outer boundary, so solidity cannot tell them apart -- fill ratio can.
    assert contour_fill_ratio(contour, solid) >= 0.99
    assert contour_fill_ratio(contour, holey) < 0.95


def test_to_builtin_leaves_plain_values_alone() -> None:
    payload = {"a": 1, "b": "text", "c": [1.5, None], "d": True}

    assert to_builtin(payload) == {"a": 1, "b": "text", "c": [1.5, None], "d": True}
