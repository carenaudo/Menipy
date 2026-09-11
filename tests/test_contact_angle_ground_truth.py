"""Contact-angle accuracy against exactly known ground truth.

Two layers, so a failure points at the right stage:

* the angle estimators alone, fed an exact analytic circular cap -- this
  isolates estimator bias from any image processing;
* the full image chain (segmentation, baseline, contacts, angle) on rendered
  caps whose angle is exact by construction.

Known limitations are marked ``xfail`` rather than omitted, so they stay visible
and flip to XPASS once fixed.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from menipy.common.auto_calibrator import AutoCalibrator
from menipy.common.geometry import circle_fit_angle_at_point, tangent_angle_at_point
from menipy.pipelines.sessile.stages import SessilePipeline

BASELINE_Y = 360.0
HALF_WIDTH = 120.0
SUBSTRATE_LINE = ((0.0, BASELINE_Y), (640.0, BASELINE_Y))


def analytic_cap(theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return an ordered closed cap contour and its left contact point.

    The arc runs from the right contact over the apex to the left contact, then
    closes along the baseline -- the same topology a segmented silhouette has.
    """
    theta = math.radians(theta_deg)
    radius = HALF_WIDTH / math.sin(theta)
    center_y = BASELINE_Y + radius * math.cos(theta)
    start = math.atan2(BASELINE_Y - center_y, HALF_WIDTH)
    end = math.atan2(BASELINE_Y - center_y, -HALF_WIDTH)
    # Image y grows downward, so the apex arc runs through decreasing angles.
    if end > start:
        end -= 2.0 * math.pi
    samples = max(int(radius * abs(end - start)), 16)
    angles = np.linspace(start, end, samples)
    arc = np.column_stack([320.0 + radius * np.cos(angles), center_y + radius * np.sin(angles)])
    closure_x = np.linspace(320.0 - HALF_WIDTH, 320.0 + HALF_WIDTH, int(2 * HALF_WIDTH))
    closure = np.column_stack([closure_x, np.full_like(closure_x, BASELINE_Y)])
    contour = np.vstack([arc, closure[1:-1]])
    return contour, np.array([320.0 - HALF_WIDTH, BASELINE_Y])


ANGLES = (20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 160)


@pytest.mark.parametrize("theta", ANGLES)
def test_circle_fit_is_exact_on_a_circular_cap(theta: int) -> None:
    contour, contact = analytic_cap(theta)
    angle, _ = circle_fit_angle_at_point(contour, contact, SUBSTRATE_LINE)
    assert angle == pytest.approx(theta, abs=0.5)


@pytest.mark.parametrize("theta", ANGLES)
def test_tangent_stays_within_its_curvature_bias(theta: int) -> None:
    """The default linear tangent reads low on curved profiles.

    Measured bias on an exact circle is 1.3-3.6 deg between 20 and 75 deg; above
    75 deg the estimator substitutes a circle fit and is exact. This pins that
    envelope so a regression in point selection -- which once produced 165 deg
    for a 45 deg cap -- cannot hide inside it.
    """
    contour, contact = analytic_cap(theta)
    angle, _ = tangent_angle_at_point(contour, contact, SUBSTRATE_LINE, 30, 2.0)
    assert angle == pytest.approx(theta, abs=4.0)


def test_flat_drop_is_measured_from_the_apex_side() -> None:
    """A closure edge just below the baseline must not flip the frame.

    On a flat drop the closure edge holds about as many vertices as the arc, and
    a median vote over vertex heights oriented the substrate normal downward,
    reporting a 45 deg cap as 165 deg.
    """
    analytic, contact = analytic_cap(45)
    # Rasterize the way segmentation does: the mask keeps a contact band below
    # the baseline, and 8-connected chain coding gives a flat arc about one
    # vertex per column -- roughly as many as its own closure edge.
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillPoly(mask, [np.round(analytic).astype(np.int32)], 255)
    band = slice(int(BASELINE_Y), int(BASELINE_Y) + 3)
    mask[band, int(320 - HALF_WIDTH) : int(320 + HALF_WIDTH) + 1] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)

    angle, _ = tangent_angle_at_point(contour, contact, SUBSTRATE_LINE, 30, 2.0)
    assert angle == pytest.approx(45, abs=4.0)


def render_cap(theta_deg: float, substrate_gray: int, path: Path) -> Path:
    """Render a dark cap of exact contact angle on a uniform substrate."""
    height, width = 480, 640
    theta = math.radians(theta_deg)
    radius = 110.0 / math.sin(theta)
    if radius * (1.0 - math.cos(theta)) > 180.0:
        radius = 180.0 / (1.0 - math.cos(theta))
    center_y = BASELINE_Y + radius * math.cos(theta)
    image = np.full((height, width), 235, np.uint8)
    image[int(BASELINE_Y):, :] = substrate_gray
    yy, xx = np.mgrid[0:height, 0:width]
    image[((xx - 320) ** 2 + (yy - center_y) ** 2 <= radius**2) & (yy < BASELINE_Y)] = 20
    image = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    return path


@pytest.fixture(scope="module")
def app_plugins(preproc_plugins) -> None:
    """Register the preprocessor plugins the application loads.

    Without them the sessile pipeline skips auto-detection entirely, which is
    not the path the GUI runs -- and the result then depended on whether some
    other test module happened to import the plugins first. The conftest
    fixture unregisters them again after this module.
    """


def measure(path: Path) -> tuple[float, float]:
    image = cv2.imread(str(path))
    calibration = AutoCalibrator(image, "sessile").detect_all()
    assert calibration.drop_contour is not None, "no drop contour detected"
    ctx = SessilePipeline().run_with_plan(
        only=["compute_metrics"],
        image=str(path),
        drop_contour=calibration.drop_contour,
        contact_points=calibration.contact_points,
        apex_point=calibration.apex_point,
        needle_rect=calibration.needle_rect,
        roi_rect=calibration.roi_rect,
        substrate_line=calibration.substrate_line,
        calibration_params={
            "needle_diameter_mm": 1.0,
            "drop_density_kg_m3": 1000.0,
            "fluid_density_kg_m3": 1.2,
        },
    )
    return ctx.results["theta_left_deg"], ctx.results["theta_right_deg"]


SUBSTRATE_DETECTION_ON_DARK = pytest.mark.xfail(
    reason="substrate detector locks onto the drop instead of a dark substrate",
    strict=False,
)


@pytest.mark.parametrize(
    ("theta", "substrate_gray"),
    [
        (45, 250),
        (60, 250),
        (90, 250),
        (105, 250),
        (120, 250),
        (135, 250),
        (150, 250),
        (120, 150),
        (135, 150),
        (150, 150),
        pytest.param(60, 150, marks=SUBSTRATE_DETECTION_ON_DARK),
        pytest.param(90, 150, marks=SUBSTRATE_DETECTION_ON_DARK),
    ],
)
def test_image_chain_recovers_the_rendered_angle(
    theta: int, substrate_gray: int, tmp_path: Path, app_plugins: None
) -> None:
    """End to end, including obtuse drops once reported as 30-60 deg.

    The tolerance guards against gross failure -- errors on these caps were
    18-123 deg before the flank, orientation and overhang fixes. The measured
    worst case since is 7.3 deg at 120 deg, where the tangent estimator fits a
    short, pixel-noisy arc; the analytic tests above bound the estimator itself.
    """
    left, right = measure(render_cap(theta, substrate_gray, tmp_path / "cap.png"))

    assert left == pytest.approx(theta, abs=8.0)
    assert right == pytest.approx(theta, abs=8.0)
