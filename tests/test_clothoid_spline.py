"""G1 clothoid-spline contact angles (linear curvature per segment)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from menipy.common import arc_spline as A
from menipy.common import clothoid_spline as C
from menipy.math.sessile_box import fit_side_box
from menipy.pipelines.sessile.metrics import compute_sessile_metrics
from tests.synthetic_sessile import (
    half_profile,
    mask_contour,
    render_image,
    sessile_case,
)


def test_hermite_clothoid_meets_both_end_conditions() -> None:
    p0 = np.array([[0.0, 0.0], [10.0, 5.0], [0.0, 0.0]])
    p1 = np.array([[30.0, 12.0], [40.0, -3.0], [50.0, 0.0]])
    th0 = np.array([0.2, -0.4, 0.6])
    th1 = np.array([0.9, 0.3, -0.6])  # the last one is a symmetric arc
    h = C.solve_hermite(p0, p1, th0, th1)
    for j in range(3):
        seg = C.ClothoidSegment(p0[j], th0[j], h.k0[j], (h.k1[j] - h.k0[j]) / h.L[j], h.L[j])
        np.testing.assert_allclose(seg.points(np.array([h.L[j]]))[0], p1[j], atol=1e-10)
        assert seg.end_heading == pytest.approx(th1[j], abs=1e-10)
    assert h.k0[2] == pytest.approx(h.k1[2])  # symmetric data: constant curvature


def test_analytic_jacobian_matches_finite_differences() -> None:
    pts, p1, p2 = sessile_case(2.0, 110.0, "gauss0.5", seed=3)
    frame = C._Frame.from_contacts(p1, p2, pts)
    edge = C._Edge(frame.to_local(pts))
    lp1, lp2 = frame.to_local(p1)[0], frame.to_local(p2)[0]
    setup = C._physics_setup(edge, lp2, 10, 0.25)
    layout, params = C._physics_params(edge, setup, (3, 2))
    params = params + np.random.default_rng(0).normal(0.0, 0.01, len(params))
    params[0] += 0.3  # off a guide vertex
    problem = C._Problem(layout, edge, lp1, lp2, g2_weight=1.0, g2_scale=50.0)
    jac = problem.jacobian(params)
    owner = problem.owner(params)
    problem.owner = lambda _p: owner
    numeric = np.empty_like(jac)
    for i in range(len(params)):
        step = np.zeros_like(params)
        step[i] = 1e-6
        problem._key = None
        up = problem.residuals(params + step).copy()
        problem._key = None
        down = problem.residuals(params - step).copy()
        numeric[:, i] = (up - down) / 2e-6
    np.testing.assert_allclose(jac, numeric, atol=1e-5)


def test_clothoids_need_fewer_segments_than_arcs() -> None:
    half = half_profile(8.0, 150.0)
    shape = fit_side_box(half[::20], *half[-1])
    assert shape.segments_needed(0.25, order=2) < shape.segments_needed(0.25, order=1)


@pytest.mark.parametrize("theta", [30.0, 90.0, 150.0])
def test_spherical_cap_is_one_segment_per_side(theta: float) -> None:
    fit = C.fit_clothoid_spline(*sessile_case(0.0, theta))
    assert fit.n_segments == 2
    assert fit.theta_p1_deg == pytest.approx(theta, abs=0.02)
    assert fit.theta_p2_deg == pytest.approx(theta, abs=0.02)


@pytest.mark.parametrize(
    ("bond", "theta", "tilt"),
    [(2.0, 90.0, 5.0), (8.0, 150.0, -3.0), (0.5, 120.0, 0.0), (8.0, 30.0, 2.0)],
)
def test_young_laplace_profiles_with_half_the_segments_of_arcs(bond, theta, tilt) -> None:
    pts, p1, p2 = sessile_case(bond, theta, tilt_deg=tilt)
    fit = C.fit_clothoid_spline(pts, p1, p2)
    arcs = A.fit_arc_spline(pts, p1, p2)
    assert fit.theta_p1_deg == pytest.approx(theta, abs=0.05)
    assert fit.theta_p2_deg == pytest.approx(theta, abs=0.05)
    assert fit.n_segments <= arcs.n_arcs


def test_noisy_edges_and_covariance_uncertainty() -> None:
    errors, sigmas = [], []
    for seed in range(4):
        pts, p1, p2 = sessile_case(2.0, 90.0, "gauss1.0", seed=seed, tilt_deg=4.0)
        fit = C.fit_clothoid_spline(pts, p1, p2)
        errors += [fit.theta_p1_deg - 90.0, fit.theta_p2_deg - 90.0]
        sigmas += [fit.sigma_p1_deg, fit.sigma_p2_deg]
    rms = float(np.sqrt(np.mean(np.square(errors))))
    assert rms < 0.8
    # the fit covariance predicts the scatter within a factor of three
    assert rms / 3.0 < float(np.mean(sigmas)) < rms * 3.0


def test_metrics_clothoid_method_with_image_refinement() -> None:
    exact, p1, p2 = sessile_case(1.0, 110.0, contact_radius_px=120.0, tilt_deg=5.0)
    image = render_image(exact, p1, p2, seed=0)
    line = ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
    metrics = compute_sessile_metrics(
        mask_contour(image), px_per_mm=100.0, substrate_line=line,
        contact_points=(tuple(p1), tuple(p2)), contact_angle_method="clothoid_spline",
        auto_detect_apex=True, image=image,
    )
    assert metrics["method"] == "clothoid_spline"
    assert metrics["method_left"] == "clothoid_spline"
    assert metrics["arc_spline"]["segment"] == "clothoid"
    assert metrics["theta_left_deg"] == pytest.approx(110.0, abs=1.0)
    assert metrics["theta_right_deg"] == pytest.approx(110.0, abs=1.0)
    json.dumps(metrics["arc_spline"])
