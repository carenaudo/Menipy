"""Unit and regression tests for Low-Bond Axisymmetric Drop Shape Analysis (LB-ADSA).

Validates:
- Pure analytical perturbation math and derivatives (Stalder et al. 2010).
- Asymptotic limits (Bo -> 0, alpha -> 0).
- Contact angle root finding across acute, perpendicular, and obtuse droplets.
- Radial least-squares profile fitting on noiseless and noisy synthetic drops.
- Surface tension estimation from Bond number and apex radius.
- Full Sessile pipeline and metrics integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from menipy.common.lbadsa_solver import fit_lbadsa_drop
from menipy.math.lbadsa import (
    cartesian_lbadsa,
    contact_angle_lbadsa,
    dradius_lbadsa,
    lbadsa_ode_profile,
    radius_lbadsa,
    surface_tension_from_bo,
    tangent_angle_lbadsa,
)
from menipy.models.geometry import SubstrateProfile
from menipy.pipelines.sessile import SessilePipeline
from menipy.pipelines.sessile.metrics import compute_sessile_metrics


class TestLBADSAMath:
    """Test pure mathematical formulations and asymptotic limits."""

    def test_apex_limits(self) -> None:
        """At alpha = 0 (apex), R(0) == R0 and dR/dalpha == 0 for any Bo."""
        for R0 in [50.0, 150.0, 300.0]:
            for Bo in [0.0, 0.05, 0.15, 0.3]:
                r_apex = radius_lbadsa(0.0, R0, Bo)
                dr_apex = dradius_lbadsa(0.0, R0, Bo)
                assert np.isclose(r_apex, R0, atol=1e-12)
                assert np.isclose(dr_apex, 0.0, atol=1e-12)

    def test_spherical_cap_limit_bo_zero(self) -> None:
        """At Bo = 0, LB-ADSA reduces identically to a spherical cap."""
        R0 = 100.0
        alphas = np.linspace(0.0, np.radians(150.0), 50)
        radii = radius_lbadsa(alphas, R0, Bo=0.0)
        dradii = dradius_lbadsa(alphas, R0, Bo=0.0)
        thetas = tangent_angle_lbadsa(alphas, R0, Bo=0.0)

        # In spherical cap, R(alpha) == R0 everywhere, dR/dalpha == 0, theta == alpha
        assert np.allclose(radii, R0, atol=1e-12)
        assert np.allclose(dradii, 0.0, atol=1e-12)
        assert np.allclose(thetas, alphas, atol=1e-12)

    def test_cartesian_coordinates_apex_origin(self) -> None:
        """Cartesian coordinates must start at apex origin (0, 0)."""
        x, z = cartesian_lbadsa(0.0, R0=120.0, Bo=0.08)
        assert np.isclose(x, 0.0, atol=1e-12)
        assert np.isclose(z, 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "theta_target_deg, Bo",
        [
            (25.0, 0.02),
            (45.0, 0.05),
            (70.0, 0.08),
            (90.0, 0.05),
            (115.0, 0.06),
            (145.0, 0.03),
        ],
    )
    def test_contact_angle_roundtrip(self, theta_target_deg: float, Bo: float) -> None:
        """contact_angle_lbadsa accurately recovers contact angle from height H."""
        R0 = 150.0
        # Determine alpha corresponding to theta_target
        alphas = np.linspace(1e-4, np.radians(160.0), 2000)
        thetas = np.degrees(tangent_angle_lbadsa(alphas, R0, Bo))
        idx = int(np.argmin(np.abs(thetas - theta_target_deg)))
        alpha_true = float(alphas[idx])
        theta_true = float(thetas[idx])

        # Compute theoretical height at this alpha
        _, z = cartesian_lbadsa(alpha_true, R0, Bo)
        height = float(z)

        recovered_theta, recovered_alpha = contact_angle_lbadsa(R0, Bo, height)
        assert np.isclose(recovered_theta, theta_true, atol=1e-4)
        assert np.isclose(recovered_alpha, alpha_true, atol=1e-4)

    def test_surface_tension_physical_values(self) -> None:
        """surface_tension_from_bo matches analytical value."""
        # Water droplet: Delta_rho = 998.2, g = 9.80665, R0 = 1.5 mm, Bo = 0.3025
        # gamma = 998.2 * 9.80665 * (1.5)^2 * 1e-3 / 0.3025 = 72.86 mN/m
        gamma = surface_tension_from_bo(Bo=0.3025, R0_mm=1.5, delta_rho=998.2, g=9.80665)
        assert np.isclose(gamma, 72.86, atol=0.1)

        # Non-physical Bo <= 0 returns NaN
        assert np.isnan(surface_tension_from_bo(Bo=0.0, R0_mm=1.5))
        assert np.isnan(surface_tension_from_bo(Bo=-0.05, R0_mm=1.5))

    def test_ode_profile_symmetry(self) -> None:
        """lbadsa_ode_profile returns symmetric coordinates centered at apex (0, 0)."""
        profile = lbadsa_ode_profile(np.array([2.5, 0.08]), physics={"g": 9.80665})
        assert profile.ndim == 2 and profile.shape[1] == 2
        # Apex is at the center
        mid = len(profile) // 2
        assert np.isclose(profile[mid, 0], 0.0, atol=1e-6)
        assert np.isclose(profile[mid, 1], 0.0, atol=1e-6)
        # Left and right x are symmetric opposites
        assert np.isclose(profile[0, 0], -profile[-1, 0], atol=1e-6)
        assert np.isclose(profile[0, 1], profile[-1, 1], atol=1e-6)


class TestLBADSASolver:
    """Test non-linear least-squares fitting of drop profiles."""

    @pytest.mark.parametrize(
        "theta_deg, Bo",
        [
            (35.0, 0.03),
            (60.0, 0.08),
            (90.0, 0.05),
            (115.0, 0.04),
            (140.0, 0.02),
        ],
    )
    def test_exact_synthetic_recovery(self, theta_deg: float, Bo: float) -> None:
        """Fit recovers noiseless synthetic parameters with near-zero residual."""
        R0 = 140.0
        alpha_max = np.radians(theta_deg)
        true_theta = float(np.degrees(tangent_angle_lbadsa(alpha_max, R0, Bo)))

        alphas = np.linspace(-alpha_max, alpha_max, 120)
        x, z = cartesian_lbadsa(np.abs(alphas), R0, Bo)
        x = np.where(alphas >= 0, x, -x)

        apex = (200.0, 60.0)
        baseline_y = 60.0 + float(np.max(z))
        contour = np.column_stack([x + apex[0], z + apex[1]])
        sub_line = ((50.0, baseline_y), (350.0, baseline_y))

        res = fit_lbadsa_drop(contour, sub_line, apex_xy=apex, optimize_bo=True)

        assert res["diagnostics"]["fit_success"] is True
        assert np.isclose(res["theta_left_deg"], true_theta, atol=0.2)
        assert np.isclose(res["theta_right_deg"], true_theta, atol=0.2)
        assert np.isclose(res["bond_number"], Bo, atol=0.01)
        assert res["rmse_px"] < 0.05

    def test_noisy_drop_robustness(self) -> None:
        """Fit converges accurately in presence of Gaussian edge jitter."""
        np.random.seed(123)
        R0 = 160.0
        Bo = 0.06
        alpha_max = np.radians(75.0)
        true_theta = float(np.degrees(tangent_angle_lbadsa(alpha_max, R0, Bo)))

        alphas = np.linspace(-alpha_max, alpha_max, 140)
        x, z = cartesian_lbadsa(np.abs(alphas), R0, Bo)
        x = np.where(alphas >= 0, x, -x)

        # Add Gaussian noise sigma = 0.5px
        x_noisy = x + np.random.normal(0, 0.5, len(x))
        z_noisy = z + np.random.normal(0, 0.5, len(z))

        apex = (250.0, 70.0)
        baseline_y = 70.0 + float(np.max(z))
        contour = np.column_stack([x_noisy + apex[0], z_noisy + apex[1]])
        sub_line = ((50.0, baseline_y), (450.0, baseline_y))

        res = fit_lbadsa_drop(contour, sub_line, apex_xy=apex, optimize_bo=True)

        assert res["diagnostics"]["fit_success"] is True
        # Angular error should be within 1 degree despite noise
        assert abs(res["theta_mean_deg"] - true_theta) < 1.0
        # RMSE should approximate the noise sigma (0.5 px)
        assert 0.3 <= res["rmse_px"] <= 0.7

    def test_curved_substrate_support(self) -> None:
        """LB-ADSA correctly accounts for curved substrate profiles."""
        R0 = 120.0
        Bo = 0.05
        alpha_max = np.radians(65.0)
        alphas = np.linspace(-alpha_max, alpha_max, 100)
        x, z = cartesian_lbadsa(np.abs(alphas), R0, Bo)
        x = np.where(alphas >= 0, x, -x)

        apex = (300.0, 100.0)
        baseline_y = 100.0 + float(np.max(z))
        contour = np.column_stack([x + apex[0], z + apex[1]])

        # Create curved arc substrate profile
        sub_prof = SubstrateProfile.from_arc((100.0, baseline_y), (300.0, baseline_y + 10.0), (500.0, baseline_y))

        res = fit_lbadsa_drop(contour, sub_prof, apex_xy=apex, optimize_bo=True)
        assert res["diagnostics"]["fit_success"] is True
        assert np.isfinite(res["theta_mean_deg"])
        assert 40.0 < res["theta_mean_deg"] < 85.0


class TestLBADSAPipelineIntegration:
    """Test sessile pipeline metrics and stages with LB-ADSA."""

    def test_compute_sessile_metrics_lbadsa(self) -> None:
        """compute_sessile_metrics computes LB-ADSA fields when requested."""
        R0 = 130.0
        Bo = 0.05
        alphas = np.linspace(-np.radians(60.0), np.radians(60.0), 80)
        x, z = cartesian_lbadsa(np.abs(alphas), R0, Bo)
        x = np.where(alphas >= 0, x, -x)

        apex = (200.0, 100.0)
        baseline_y = 100.0 + float(np.max(z))
        contour = np.column_stack([x + apex[0], z + apex[1]])
        sub_line = ((50.0, baseline_y), (350.0, baseline_y))

        metrics = compute_sessile_metrics(
            contour,
            px_per_mm=10.0,
            substrate_line=sub_line,
            apex=apex,
            contact_angle_method="lbadsa",
        )

        assert metrics["method"] == "lbadsa"
        assert "bond_number" in metrics
        assert "surface_tension_mN_m" in metrics
        assert "lbadsa_diagnostics" in metrics
        assert "lbadsa_model_contour_xy" in metrics
        assert np.isclose(metrics["theta_left_deg"], 60.5, atol=1.0)
        assert np.isclose(metrics["bond_number"], 0.05, atol=0.01)

    def test_sessile_pipeline_execution_with_lbadsa(self) -> None:
        """SessilePipeline runs full stage sequence with solver_name='lbadsa'."""
        R0 = 150.0
        Bo = 0.06
        alphas = np.linspace(-np.radians(50.0), np.radians(50.0), 100)
        x, z = cartesian_lbadsa(np.abs(alphas), R0, Bo)
        x = np.where(alphas >= 0, x, -x)

        # Create binary synthetic drop image
        img = np.zeros((400, 500), dtype=np.uint8)
        import cv2

        apex_pt = (250, 100)
        baseline_y = 100 + int(np.max(z))
        pts = np.column_stack([x + apex_pt[0], z + apex_pt[1]]).astype(np.int32)
        # Close contour with baseline
        poly = np.vstack([pts, [[pts[0, 0], baseline_y]]])
        cv2.fillPoly(img, [poly], 255)

        pipe = SessilePipeline()
        pipe.solver_name = "lbadsa"
        ctx = pipe.run(image=img, analysis_params={"contact_angle_method": "lbadsa"})

        assert ctx.results is not None
        assert "contact_angle_deg" in ctx.results
        assert "theta_left_deg" in ctx.results
        assert "theta_right_deg" in ctx.results
        assert ctx.results["method"] == "lbadsa"
        assert ctx.error is None
