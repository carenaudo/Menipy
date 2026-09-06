"""Unit and parametric tests for the Active Contour (Snake) mathematical engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from menipy.math.active_contour import (
    ActiveContourConfig,
    BSplineSnakeResult,
    SnakeBoundaryCondition,
    build_pentadiagonal_matrix,
    compute_contour_normals,
    compute_substrate_angle,
    evolve_active_contour,
    fit_bspline_snake,
    project_point_to_line,
    resample_contour_arclength,
)

# -----------------------------------------------------------------------------
# Test Fixtures & Synthetic Image Generators
# -----------------------------------------------------------------------------


def make_synthetic_circle_image(
    size: tuple[int, int] = (120, 120),
    center: tuple[float, float] = (60.0, 60.0),
    radius: float = 30.0,
    intensity_inside: int = 40,
    intensity_outside: int = 220,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """Create a synthetic image with a clean circular disk."""
    h, w = size
    img = np.full((h, w), intensity_outside, dtype=np.uint8)
    cv2.circle(img, (int(round(center[0])), int(round(center[1]))), int(round(radius)), intensity_inside, -1)
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float64)
        img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return img


def make_synthetic_sessile_image(
    size: tuple[int, int] = (140, 160),
    center_x: float = 80.0,
    substrate_y: float = 100.0,
    radius: float = 35.0,
    theta_deg: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic sessile droplet image and its exact theoretical boundary.

    droplet apex is at (center_x, substrate_y - h).
    """
    h_img, w_img = size
    img = np.full((h_img, w_img), 230, dtype=np.uint8)

    # Circle center calculation for contact angle theta
    # Dome apex is at center_y - radius, substrate is at substrate_y
    # So center_y = substrate_y + radius * cos(theta)
    theta_rad = np.radians(theta_deg)
    center_y = substrate_y + radius * np.cos(theta_rad)

    # Draw droplet disk and clip below substrate
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.circle(mask, (int(round(center_x)), int(round(center_y))), int(round(radius)), 255, -1)
    # Clear below substrate
    mask[int(round(substrate_y)) :, :] = 0
    img[mask == 255] = 40

    # Draw substrate line
    cv2.line(img, (0, int(round(substrate_y))), (w_img, int(round(substrate_y))), 100, 1)

    # Generate theoretical boundary points from left contact to right contact
    angles = np.linspace(-theta_rad, theta_rad, 60)
    true_x = center_x + radius * np.sin(angles)
    true_y = center_y - radius * np.cos(angles)
    true_contour = np.column_stack([true_x, true_y])

    return img, true_contour


# -----------------------------------------------------------------------------
# 1. Matrix Construction & Boundary Condition Tests
# -----------------------------------------------------------------------------


class TestMatrixConstruction:
    """Tests for the pentadiagonal shape matrix A."""

    @pytest.mark.parametrize("bc", list(SnakeBoundaryCondition))
    def test_matrix_dimensions_and_invertibility(self, bc: SnakeBoundaryCondition):
        n = 40
        alpha = 0.02
        beta = 0.15
        gamma = 0.01

        A = build_pentadiagonal_matrix(n, alpha, beta, bc)
        assert A.shape == (n, n)
        assert np.all(np.isfinite(A))

        # Matrix (A + gamma * I) must be well-conditioned and invertible
        M = A + gamma * np.eye(n)
        cond = np.linalg.cond(M)
        assert cond < 1e7, f"Condition number too high: {cond}"

        inv_M = np.linalg.inv(M)
        assert np.all(np.isfinite(inv_M))
        identity_check = inv_M @ M
        np.testing.assert_allclose(identity_check, np.eye(n), atol=1e-10)

    def test_periodic_matrix_circulant(self):
        n = 20
        A = build_pentadiagonal_matrix(n, 0.01, 0.1, SnakeBoundaryCondition.PERIODIC)
        # Periodic matrix should have equal diagonal elements
        diag = np.diag(A)
        assert np.allclose(diag, diag[0])
        # Check that top row wraps around to the last columns
        assert abs(A[0, -1]) > 0.0
        assert abs(A[0, -2]) > 0.0


# -----------------------------------------------------------------------------
# 2. Line Projection & Substrate Angle Tests
# -----------------------------------------------------------------------------


class TestLineProjection:
    """Tests for orthogonal line projections and angle calculations."""

    def test_project_horizontal_scalar(self):
        pt = (25.0, 80.0)
        proj = project_point_to_line(pt, 120.0)
        assert proj[0] == pytest.approx(25.0)
        assert proj[1] == pytest.approx(120.0)

    def test_project_two_points_tilted(self):
        line = ((0.0, 0.0), (100.0, 100.0))  # 45-degree line y = x
        pt = (10.0, 20.0)
        proj = project_point_to_line(pt, line)
        # Projection of (10, 20) onto y=x is (15, 15)
        assert proj[0] == pytest.approx(15.0)
        assert proj[1] == pytest.approx(15.0)

    def test_project_line_coefficients(self):
        # 3x + 4y - 25 = 0
        line = (3.0, 4.0, -25.0)
        pt = (0.0, 0.0)
        proj = project_point_to_line(pt, line)
        # Distance = -25/25 = -1, proj = - (3*-1, 4*-1) = (3, 4)
        assert proj[0] == pytest.approx(3.0)
        assert proj[1] == pytest.approx(4.0)
        # Verify it lies on the line: 3*3 + 4*4 - 25 = 9 + 16 - 25 = 0
        assert 3.0 * proj[0] + 4.0 * proj[1] - 25.0 == pytest.approx(0.0, abs=1e-12)

    def test_compute_substrate_angle(self):
        assert compute_substrate_angle(100.0) == 0.0
        assert compute_substrate_angle(((0.0, 0.0), (100.0, 0.0))) == 0.0
        # 45 deg line
        assert compute_substrate_angle(((0.0, 0.0), (50.0, 50.0))) == pytest.approx(45.0)
        # -30 deg line
        rad = np.radians(-30.0)
        assert compute_substrate_angle(((0.0, 0.0), (100.0, 100.0 * np.tan(rad)))) == pytest.approx(-30.0)


# -----------------------------------------------------------------------------
# 3. Arc-Length Resampling & Geometry Tests
# -----------------------------------------------------------------------------


class TestContourGeometry:
    """Tests for arc-length resampling, normal vectors, and curvatures."""

    def test_resample_contour_arclength_uniformity(self):
        # Create non-uniform points along a line
        t = np.array([0.0, 0.05, 0.1, 0.5, 0.9, 1.0])
        xy = np.column_stack([t * 100.0, np.zeros_like(t)])

        resampled = resample_contour_arclength(xy, n_points=11, closed=False, preserve_endpoints=True)
        assert len(resampled) == 11
        # Check endpoints preserved
        np.testing.assert_allclose(resampled[0], [0.0, 0.0])
        np.testing.assert_allclose(resampled[-1], [100.0, 0.0])

        # Step size should be exactly 10.0
        diffs = np.diff(resampled, axis=0)
        step_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        np.testing.assert_allclose(step_lengths, 10.0, atol=1e-9)

    def test_compute_contour_normals_circle(self):
        r = 25.0
        angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
        xy = np.column_stack([50.0 + r * np.cos(angles), 50.0 + r * np.sin(angles)])

        tangents, normals, curvatures = compute_contour_normals(xy, closed=True)

        # Normals must have unit length
        norm_lengths = np.hypot(normals[:, 0], normals[:, 1])
        np.testing.assert_allclose(norm_lengths, 1.0, atol=1e-7)

        # Tangents and normals must be orthogonal
        dots = tangents[:, 0] * normals[:, 0] + tangents[:, 1] * normals[:, 1]
        np.testing.assert_allclose(dots, 0.0, atol=1e-7)

        # Curvature on a circle of radius r must be ~ 1/r
        # (ignoring discretization boundary variation)
        np.testing.assert_allclose(np.abs(curvatures), 1.0 / r, rtol=0.05)


# -----------------------------------------------------------------------------
# 4. Periodic Active Contour Evolution (Pendant / Free Drop)
# -----------------------------------------------------------------------------


class TestPeriodicSnake:
    """Tests for closed-loop active contour snapping."""

    def test_snaps_to_circular_silhouette(self):
        true_r = 30.0
        center = (60.0, 60.0)
        img = make_synthetic_circle_image(size=(120, 120), center=center, radius=true_r)

        # Initialize as slightly larger circle (radius 36)
        init_r = 36.0
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        init_xy = np.column_stack([center[0] + init_r * np.cos(angles), center[1] + init_r * np.sin(angles)])

        cfg = ActiveContourConfig(
            alpha=0.01,
            beta=0.1,
            gamma=0.01,
            w_edge=1.0,
            gaussian_sigma=2.0,
            max_iterations=200,
            convergence=0.02,
        )

        res = evolve_active_contour(img, init_xy, config=cfg, boundary_condition=SnakeBoundaryCondition.PERIODIC)

        assert res.converged
        # Calculate radial distances of converged snake from center
        dists = np.hypot(res.xy[:, 0] - center[0], res.xy[:, 1] - center[1])
        mean_r = float(np.mean(dists))
        # Mean radius should match true radius within 0.5 px
        assert abs(mean_r - true_r) < 0.5, f"Converged mean radius {mean_r} != {true_r}"


# -----------------------------------------------------------------------------
# 5. Substrate Sliding Constraint (Sessile Droplet)
# -----------------------------------------------------------------------------


class TestSubstrateSlidingSnake:
    """Tests for open active contour with sliding substrate line constraint."""

    def test_endpoints_strictly_adhere_to_substrate(self):
        sub_y = 100.0
        img, true_contour = make_synthetic_sessile_image(
            size=(140, 160),
            center_x=80.0,
            substrate_y=sub_y,
            radius=35.0,
            theta_deg=65.0,
        )

        # Initialize with perturbed contour (expanded by 5 px)
        init_xy = true_contour.copy()
        init_xy[1:-1, 1] -= 5.0  # Lift dome upwards
        init_xy[0, 0] -= 4.0  # Shift left contact point
        init_xy[-1, 0] += 4.0  # Shift right contact point

        cfg = ActiveContourConfig(
            alpha=0.02,
            beta=0.1,
            gamma=0.01,
            w_edge=1.0,
            gaussian_sigma=2.0,
            max_iterations=150,
            resample_interval=10,
            convergence=0.03,
        )

        res = evolve_active_contour(
            img,
            init_xy,
            config=cfg,
            boundary_condition=SnakeBoundaryCondition.SLIDING_LINE,
            substrate_line=sub_y,
        )

        # INVARIANCE CRITERION: Both endpoints MUST lie exactly on substrate line
        assert res.xy[0, 1] == pytest.approx(sub_y, abs=1e-10)
        assert res.xy[-1, 1] == pytest.approx(sub_y, abs=1e-10)

        # Apex should accurately track the true dome (apex is min Y)
        true_apex_y = float(np.min(true_contour[:, 1]))
        res_apex_y = float(np.min(res.xy[:, 1]))
        assert abs(res_apex_y - true_apex_y) < 1.0


# -----------------------------------------------------------------------------
# 6. Pinned Boundary Condition (Needle Cannula)
# -----------------------------------------------------------------------------


class TestPinnedSnake:
    """Tests for open active contour with fixed pinned ends."""

    def test_pinned_endpoints_remain_invariant(self):
        img = np.full((100, 100), 200, dtype=np.uint8)
        pinned = ((30.0, 20.0), (70.0, 20.0))

        init_x = np.linspace(30.0, 70.0, 20)
        init_y = 20.0 + 30.0 * np.sin(np.linspace(0, np.pi, 20))
        init_xy = np.column_stack([init_x, init_y])

        cfg = ActiveContourConfig(max_iterations=50)

        res = evolve_active_contour(
            img,
            init_xy,
            config=cfg,
            boundary_condition=SnakeBoundaryCondition.PINNED,
            pinned_endpoints=pinned,
        )

        # Endpoints must remain fixed
        np.testing.assert_allclose(res.xy[0], pinned[0], atol=1e-12)
        np.testing.assert_allclose(res.xy[-1], pinned[1], atol=1e-12)


# -----------------------------------------------------------------------------
# 7. Parametric B-Spline Snake & DropSnake Contact Angles
# -----------------------------------------------------------------------------


class TestBSplineSnake:
    """Tests for continuous parametric B-spline fitting and analytical derivatives."""

    @pytest.mark.parametrize("target_theta", [35.0, 60.0, 90.0, 120.0, 145.0])
    def test_contact_angle_analytical_accuracy(self, target_theta: float):
        sub_y = 120.0
        _, true_contour = make_synthetic_sessile_image(
            size=(160, 180),
            center_x=90.0,
            substrate_y=sub_y,
            radius=40.0,
            theta_deg=target_theta,
        )

        bspline_res = fit_bspline_snake(
            true_contour,
            substrate_line=sub_y,
            num_eval_points=100,
            smoothing=0.0,
        )

        assert isinstance(bspline_res, BSplineSnakeResult)
        assert bspline_res.contact_angles_deg is not None
        theta_left, theta_right = bspline_res.contact_angles_deg

        # Analytical B-spline tangent must recover the exact geometric angle within 1.0 deg
        assert abs(theta_left - target_theta) < 1.0, f"Left: {theta_left} != {target_theta}"
        assert abs(theta_right - target_theta) < 1.0, f"Right: {theta_right} != {target_theta}"

    def test_tilted_substrate_contact_angle(self):
        # 10 degree tilted substrate
        tilt_deg = 10.0
        tilt_rad = np.radians(tilt_deg)
        theta_deg = 70.0
        theta_rad = np.radians(theta_deg)
        radius = 40.0

        u_tan = np.array([np.cos(tilt_rad), np.sin(tilt_rad)])
        u_norm = np.array([np.sin(tilt_rad), -np.cos(tilt_rad)])

        # Base point on substrate and drop center
        p_base = np.array([90.0, 100.0])
        center = p_base - radius * np.cos(theta_rad) * u_norm
        sub_line = (tuple(p_base - 50.0 * u_tan), tuple(p_base + 50.0 * u_tan))

        angles = np.linspace(-theta_rad, theta_rad, 60)
        tilted_contour = np.array([center + radius * np.sin(a) * u_tan + radius * np.cos(a) * u_norm for a in angles])

        bspline_res = fit_bspline_snake(
            tilted_contour,
            substrate_line=sub_line,
            num_eval_points=100,
        )
        assert bspline_res.contact_angles_deg is not None
        theta_left, theta_right = bspline_res.contact_angles_deg
        assert abs(theta_left - theta_deg) < 1.0, f"Left: {theta_left} != {theta_deg}"
        assert abs(theta_right - theta_deg) < 1.0, f"Right: {theta_right} != {theta_deg}"
