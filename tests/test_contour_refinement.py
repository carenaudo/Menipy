"""Tests for the Contour Refinement stage in Menipy."""

from __future__ import annotations

import cv2
import numpy as np

from menipy.common.contour_refinement import refine_contour
from menipy.models.config import ContourSmoothingSettings
from menipy.models.context import Context
from menipy.models.geometry import Contour
from menipy.pipelines.sessile.stages import SessilePipeline


def make_synthetic_sessile_image(
    size: tuple[int, int] = (160, 200),
    center_x: float = 100.0,
    substrate_y: float = 110.0,
    radius: float = 45.0,
    theta_deg: float = 60.0,
    noise_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic sessile droplet image and its exact theoretical boundary.

    droplet apex is at (center_x, substrate_y - h).
    """
    h_img, w_img = size
    img = np.full((h_img, w_img), 230, dtype=np.uint8)

    theta_rad = np.radians(theta_deg)
    center_y = substrate_y + radius * np.cos(theta_rad)

    # Draw droplet disk and clip below substrate
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(center_x)), int(round(center_y))),
        int(round(radius)),
        255,
        -1,
    )
    mask[int(round(substrate_y)) :, :] = 0
    img[mask == 255] = 40

    # Draw substrate line
    cv2.line(img, (0, int(round(substrate_y))), (w_img, int(round(substrate_y))), 100, 1)

    if noise_sigma > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_sigma, img.shape).astype(np.float64)
        img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    # Generate theoretical boundary points from left contact to right contact
    angles = np.linspace(-theta_rad, theta_rad, 60)
    true_x = center_x + radius * np.sin(angles)
    true_y = center_y - radius * np.cos(angles)
    true_contour = np.column_stack([true_x, true_y])

    return img, true_contour


# -----------------------------------------------------------------------------
# Unit Tests for refine_contour
# -----------------------------------------------------------------------------


def test_contour_refinement_disabled():
    """Disabled settings should return context unchanged."""
    ctx = Context()
    pts = np.array([[10.0, 50.0], [20.0, 30.0], [30.0, 50.0]])
    ctx.contour = Contour(xy=pts)
    settings = ContourSmoothingSettings(enabled=False)

    ctx_out = refine_contour(ctx, settings)
    assert ctx_out is ctx
    assert np.allclose(ctx.contour.xy, pts)
    assert ctx.smoothing_results is None


def test_contour_refinement_empty_contour():
    """Empty or missing contour should log warning and return cleanly."""
    ctx = Context()
    settings = ContourSmoothingSettings(enabled=True)
    ctx_out = refine_contour(ctx, settings)
    assert ctx_out is ctx
    assert ctx.smoothing_results is None


def test_active_contour_refinement_acute_droplet():
    """Refining an acute droplet with active contour should converge and respect substrate line."""
    substrate_y = 110.0
    img, true_contour = make_synthetic_sessile_image(
        theta_deg=60.0, substrate_y=substrate_y, noise_sigma=1.0
    )

    # Perturb initial contour slightly upward into droplet
    init_contour = true_contour.copy()
    init_contour[1:-1, 1] -= 1.5

    ctx = Context()
    ctx.image = img
    ctx.contour = Contour(xy=init_contour)
    ctx.substrate_line = ((0.0, substrate_y), (200.0, substrate_y))

    settings = ContourSmoothingSettings(
        enabled=True,
        method="active_contour",
        snake_alpha=0.05,
        snake_beta=0.1,
        snake_max_iterations=80,
    )

    ctx = refine_contour(ctx, settings)

    assert ctx.contour is not None
    assert ctx.sessile_calc_contour is not None
    assert ctx.smoothing_results is not None
    assert ctx.smoothing_results["method"] == "active_contour"

    # Endpoints strictly on substrate
    refined_pts = np.asarray(ctx.contour.xy)
    assert abs(refined_pts[0, 1] - substrate_y) < 1e-3
    assert abs(refined_pts[-1, 1] - substrate_y) < 1e-3

    # Contact angles should be close to theoretical 60 deg
    left_ang = ctx.smoothing_results["left_angle_deg"]
    right_ang = ctx.smoothing_results["right_angle_deg"]
    assert 50.0 < left_ang < 70.0
    assert 50.0 < right_ang < 70.0


def test_active_contour_refinement_obtuse_droplet():
    """Refining an obtuse droplet (>90 deg) should not scramble coordinates."""
    substrate_y = 100.0
    img, true_contour = make_synthetic_sessile_image(
        theta_deg=125.0, substrate_y=substrate_y, noise_sigma=0.5
    )

    # Initial contour slightly perturbed
    init_contour = true_contour.copy()
    init_contour[1:-1, 1] -= 1.0

    ctx = Context()
    ctx.image = img
    ctx.contour = Contour(xy=init_contour)
    ctx.substrate_line = ((0.0, substrate_y), (200.0, substrate_y))

    settings = ContourSmoothingSettings(
        enabled=True,
        method="active_contour",
        snake_alpha=0.05,
        snake_beta=0.1,
        snake_max_iterations=80,
    )

    ctx = refine_contour(ctx, settings)

    assert ctx.contour is not None
    refined_pts = np.asarray(ctx.contour.xy)

    # Check that points are ordered consecutively along the curve without scrambling
    step_dists = np.linalg.norm(np.diff(refined_pts, axis=0), axis=1)
    assert np.all(step_dists < 10.0)

    # Endpoints on substrate
    assert abs(refined_pts[0, 1] - substrate_y) < 1e-3
    assert abs(refined_pts[-1, 1] - substrate_y) < 1e-3

    # Contact angles should be obtuse (>90 deg, ~125 deg)
    left_ang = ctx.smoothing_results["left_angle_deg"]
    right_ang = ctx.smoothing_results["right_angle_deg"]
    assert left_ang > 90.0
    assert right_ang > 90.0
    assert 105.0 < left_ang < 140.0
    assert 105.0 < right_ang < 140.0


def test_bspline_refinement_without_image():
    """B-spline refinement should work purely geometrically without an image."""
    substrate_y = 110.0
    _, true_contour = make_synthetic_sessile_image(
        theta_deg=55.0, substrate_y=substrate_y
    )

    ctx = Context()
    ctx.contour = Contour(xy=true_contour)
    ctx.substrate_line = ((0.0, substrate_y), (200.0, substrate_y))

    settings = ContourSmoothingSettings(
        enabled=True,
        method="bspline",
        spline_eval_points=120,
    )

    ctx = refine_contour(ctx, settings)

    assert ctx.contour is not None
    assert len(ctx.contour.xy) == 120
    assert ctx.smoothing_results["method"] == "bspline"
    assert "left_angle_deg" in ctx.smoothing_results
    assert "right_angle_deg" in ctx.smoothing_results
    assert 48.0 < ctx.smoothing_results["left_angle_deg"] < 62.0


def test_active_contour_fallback_to_bspline_when_no_image():
    """Active contour method should automatically fall back to B-spline when no image is present."""
    pts = np.array([
        [80.0, 100.0],
        [85.0, 80.0],
        [100.0, 60.0],
        [115.0, 80.0],
        [120.0, 100.0],
    ])

    ctx = Context()
    ctx.contour = Contour(xy=pts)
    ctx.substrate_line = ((0.0, 100.0), (200.0, 100.0))

    settings = ContourSmoothingSettings(
        enabled=True,
        method="active_contour",
    )

    ctx = refine_contour(ctx, settings)

    assert ctx.contour is not None
    assert ctx.smoothing_results["fallback"] == "bspline"


def test_savgol_method_acute_droplet():
    """Legacy Savitzky-Golay method should work on acute droplets."""
    pts = np.array([
        [80.0, 100.0],
        [85.0, 85.0],
        [90.0, 75.0],
        [100.0, 65.0],
        [110.0, 75.0],
        [115.0, 85.0],
        [120.0, 100.0],
    ])

    ctx = Context()
    ctx.contour = Contour(xy=pts)
    ctx.substrate_line = ((0.0, 100.0), (200.0, 100.0))

    settings = ContourSmoothingSettings(
        enabled=True,
        method="savgol",
        window_length=5,
        polyorder=2,
    )

    ctx = refine_contour(ctx, settings)

    assert ctx.contour is not None
    assert ctx.smoothing_results["method"] == "savgol"
    assert "left_angle_deg" in ctx.smoothing_results


def test_sessile_pipeline_do_contour_refinement_integration():
    """Pipeline stage do_contour_refinement should execute contour refinement seamlessly."""
    substrate_y = 110.0
    img, true_contour = make_synthetic_sessile_image(
        theta_deg=65.0, substrate_y=substrate_y
    )

    # Initial contour
    init_contour = true_contour.copy()
    init_contour[1:-1, 1] -= 1.0

    pipeline = SessilePipeline()
    ctx = Context()
    ctx.image = img
    ctx.contour = Contour(xy=init_contour)
    ctx.substrate_line = ((0.0, substrate_y), (200.0, substrate_y))
    ctx.contour_smoothing_settings = ContourSmoothingSettings(
        enabled=True,
        method="active_contour",
        snake_max_iterations=60,
    )

    # Run do_contour_refinement
    ctx = pipeline.do_contour_refinement(ctx)
    assert ctx is not None
    assert ctx.sessile_calc_contour is not None
    assert ctx.sessile_calc_contact_points is not None
    assert ctx.contact_points is not None
    assert ctx.smoothing_results is not None

    # Verify downstream do_geometric_features executes cleanly with refined contour
    ctx = pipeline.do_geometric_features(ctx)
    assert ctx is not None
    assert ctx.geometry is not None
    assert ctx.geometry.apex_xy is not None
