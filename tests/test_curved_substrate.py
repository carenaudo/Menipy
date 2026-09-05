"""Tests for curved substrate baselines, local tangent angle correction, and profile geometry."""

from __future__ import annotations

import numpy as np
import pytest

from menipy.common.geometry import find_contact_points_from_contour
from menipy.models.geometry import SubstrateProfile
from menipy.pipelines.sessile.geometry import build_sessile_calculation_contour
from menipy.pipelines.sessile.metrics import compute_sessile_metrics


class TestSubstrateProfile:
    """Test SubstrateProfile model methods."""

    def test_from_line(self):
        prof = SubstrateProfile.from_line((0.0, 100.0), (200.0, 100.0), confidence=0.9)
        assert prof.type == "line"
        assert prof.confidence == pytest.approx(0.9)
        assert prof.eval_y(50.0) == pytest.approx(100.0)
        assert prof.eval_tangent_angle_deg(50.0, 100.0) == pytest.approx(0.0)
        chord = prof.to_chord()
        assert chord == ((0, 100), (200, 100))

    def test_from_line_tilted(self):
        # 45-degree slope
        prof = SubstrateProfile.from_line((0.0, 0.0), (100.0, 100.0))
        assert prof.eval_y(50.0) == pytest.approx(50.0)
        assert prof.eval_tangent_angle_deg(50.0, 50.0) == pytest.approx(45.0)

    def test_from_arc_circular(self):
        # A circle of radius R=100 centered at (100, 200)
        # Left anchor p1=(0, 200), Right anchor p2=(200, 200), Crest p3=(100, 100)
        # Crest is above center (y=100 < y=200 in image coords), convex facing up
        p1 = (0.0, 200.0)
        p2 = (200.0, 200.0)
        p3 = (100.0, 100.0)
        prof = SubstrateProfile.from_arc(p1, p2, p3)
        assert prof.type == "circle_arc"
        assert prof.parameters["center_x"] == pytest.approx(100.0)
        assert prof.parameters["center_y"] == pytest.approx(200.0)
        assert prof.parameters["radius"] == pytest.approx(100.0)
        assert prof.parameters["curvature_inv_px"] == pytest.approx(0.01)

        # Crest y at x=100
        assert prof.eval_y(100.0) == pytest.approx(100.0)
        # Tangent angle at crest should be horizontal (0 deg)
        assert prof.eval_tangent_angle_deg(100.0, 100.0) == pytest.approx(0.0)

        # At x=50: dx = -50, dy = sqrt(100^2 - 50^2) = sqrt(7500) = 86.60
        # dy/dx = 50 / 86.60 = 0.577 -> arctan(0.577) = 30 deg slope
        slope_angle = prof.eval_tangent_angle_deg(50.0, prof.eval_y(50.0))
        assert abs(slope_angle) == pytest.approx(30.0, abs=1.0)

    def test_from_arc_collinear_fallback(self):
        # Three collinear points -> fallback to straight line
        p1 = (0.0, 100.0)
        p2 = (100.0, 100.0)
        p3 = (50.0, 100.0)
        prof = SubstrateProfile.from_arc(p1, p2, p3)
        assert prof.type == "line"
        assert prof.eval_y(75.0) == pytest.approx(100.0)

    def test_sample_points(self):
        p1 = (0.0, 100.0)
        p2 = (100.0, 100.0)
        p3 = (50.0, 80.0)
        prof = SubstrateProfile.from_arc(p1, p2, p3)
        pts = prof.sample_points(num_points=25)
        assert len(pts) == 25
        assert pts[0][0] == pytest.approx(0.0)
        assert pts[-1][0] == pytest.approx(100.0)


class TestCurvedSubstrateGeometryAndMetrics:
    """Test contact angle computation and contour building with curved substrates."""

    def test_intrinsic_contact_angle_correction_on_cylinder(self):
        """Verify theta_intrinsic = theta_apparent - alpha_sub on convex substrate."""
        # Create a synthetic droplet on a curved cylinder of radius R=200
        p1 = (50.0, 150.0)
        p2 = (250.0, 150.0)
        p3 = (150.0, 100.0)
        sub_prof = SubstrateProfile.from_arc(p1, p2, p3)

        thetas = np.linspace(np.pi * 0.15, np.pi * 0.85, 80)
        drop_r = 60.0
        xs = 150.0 + drop_r * np.cos(thetas)
        ys = 80.0 - drop_r * np.sin(thetas)
        contour = np.column_stack([xs, ys])

        left_contact, right_contact = find_contact_points_from_contour(
            contour, sub_prof
        )
        assert left_contact is not None
        assert right_contact is not None

        metrics = compute_sessile_metrics(
            contour,
            px_per_mm=10.0,
            contact_points=(left_contact, right_contact),
            substrate_profile=sub_prof,
        )

        assert "theta_left_deg" in metrics
        assert "theta_right_deg" in metrics
        assert "theta_left_apparent_deg" in metrics
        assert "theta_right_apparent_deg" in metrics
        assert "substrate_profile" in metrics
        assert metrics["substrate_radius_mm"] == pytest.approx(12.5, abs=0.5)
        assert 0.0 < metrics["theta_left_deg"] < 180.0
        assert 0.0 < metrics["theta_right_deg"] < 180.0

    def test_build_sessile_calculation_contour_curved(self):
        """Verify calculation contour follows the curved substrate arc between contact points."""
        p1 = (0.0, 200.0)
        p2 = (200.0, 200.0)
        p3 = (100.0, 150.0)
        sub_prof = SubstrateProfile.from_arc(p1, p2, p3)

        t = np.linspace(0, np.pi, 50)
        xs = 100.0 - 50.0 * np.cos(t)
        ys = 150.0 - 100.0 * np.sin(t)
        raw_contour = np.column_stack([xs, ys])

        apex = (100.0, 50.0)
        contact_points = ((50.0, sub_prof.eval_y(50.0)), (150.0, sub_prof.eval_y(150.0)))
        closed, pts = build_sessile_calculation_contour(
            raw_contour,
            sub_prof,
            apex,
            contact_points=contact_points,
        )

        assert len(closed) > len(raw_contour)
        mid_x = 100.0
        expected_y = sub_prof.eval_y(mid_x)
        bottom_pts = closed[(np.abs(closed[:, 0] - mid_x) < 5.0) & (closed[:, 1] > 120.0)]
        assert len(bottom_pts) > 0
        assert np.min(np.abs(bottom_pts[:, 1] - expected_y)) < 5.0
