"""Apex-anchored G1 arc-spline contact angles.

Ground truth comes from independent Young-Laplace integrations
(:mod:`tests.synthetic_sessile`) and anti-aliased renderings of them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from menipy.common import arc_spline as A
from menipy.common.geometry import tangent_angle_at_point
from menipy.pipelines.sessile.metrics import compute_sessile_metrics
from tests.synthetic_sessile import (
    closed_silhouette,
    half_profile,
    mask_contour,
    render_image,
    sessile_case,
)


def test_arc_through_passes_through_its_end_point() -> None:
    start, end = np.array([0.0, 0.0]), np.array([30.0, 12.0])
    k, length, end_heading = A.arc_through(start, 0.2, end)
    arc = A.ArcSegment(start, 0.2, k, length)
    np.testing.assert_allclose(arc.points(np.array([length]))[0], end, atol=1e-9)
    assert arc.end_heading == pytest.approx(end_heading)


@pytest.mark.parametrize("slide_px", [0.0, 5.0])
def test_analytic_jacobian_matches_finite_differences(slide_px: float) -> None:
    pts, p1, p2 = sessile_case(2.0, 110.0, "gauss0.5", seed=3)
    frame = A._Frame.from_contacts(p1, p2, pts)
    edge = A._Edge(frame.to_local(pts))
    lp1, lp2 = frame.to_local(p1)[0], frame.to_local(p2)[0]
    model = A._Model(2, 3, slide_px)
    ta = edge.seg[np.argmin(edge.q[:, 1])] + 0.37  # off a vertex: the guide is piecewise linear
    params = model.pack(
        ta, 0.7, np.array([0.3, 0.7]) * ta, np.array([1.0, -0.5]),
        ta + np.array([0.2, 0.5, 0.8]) * (edge.total - ta), np.array([0.3, 0.0, -1.2]),
        (1.3, -2.1),
    )
    problem = A._Problem(model, edge, lp1, lp2)
    jac = problem.jacobian(params)
    owner = problem.owner(params)
    problem.owner = lambda _p: owner  # ownership is piecewise constant by design
    numeric = np.empty_like(jac)
    for i in range(len(params)):
        step = np.zeros_like(params)
        step[i] = 1e-6
        problem._key = None
        up = problem.residuals(params + step).copy()
        problem._key = None
        down = problem.residuals(params - step).copy()
        numeric[:, i] = (up - down) / 2e-6
    np.testing.assert_allclose(jac, numeric, atol=1e-6)


@pytest.mark.parametrize("init", ["physics", "blind"])
@pytest.mark.parametrize("theta", [30.0, 90.0, 150.0])
def test_spherical_cap_is_two_exact_arcs(init: str, theta: float) -> None:
    pts, p1, p2 = sessile_case(0.0, theta)
    fit = A.fit_arc_spline(pts, p1, p2, init=init)
    assert fit.n_arcs == 2
    assert fit.theta_p1_deg == pytest.approx(theta, abs=0.02)
    assert fit.theta_p2_deg == pytest.approx(theta, abs=0.02)


@pytest.mark.parametrize(
    ("bond", "theta", "tilt"),
    [(0.5, 90.0, 0.0), (2.0, 90.0, 7.0), (2.0, 120.0, -5.0), (8.0, 150.0, 3.0), (8.0, 30.0, 0.0)],
)
def test_young_laplace_profiles_are_recovered_after_bias_correction(bond, theta, tilt) -> None:
    pts, p1, p2 = sessile_case(bond, theta, tilt_deg=tilt)
    fit = A.fit_arc_spline(pts, p1, p2)
    assert fit.init == "physics"
    assert fit.theta_p1_deg == pytest.approx(theta, abs=0.1)
    assert fit.theta_p2_deg == pytest.approx(theta, abs=0.1)
    # the discretization bias is real and was removed, not absent
    assert abs(fit.correction_p1_deg) > 0.05 or bond < 1.0
    assert fit.physics["p2"]["theta_deg"] == pytest.approx(theta, abs=0.2)


def test_arcs_follow_the_physics_prediction_of_curvature() -> None:
    sphere = A.fit_arc_spline(*sessile_case(0.0, 90.0))
    flat = A.fit_arc_spline(*sessile_case(8.0, 150.0))
    assert sphere.n_arcs == 2
    assert flat.n_arcs > 6


@pytest.mark.parametrize(
    ("bond", "theta", "noise"),
    [(2.0, 90.0, "gauss1.0"), (2.0, 120.0, "gauss0.5"), (0.5, 60.0, "quantized"), (8.0, 150.0, "gauss0.5")],
)
def test_noisy_edges_stay_within_a_degree(bond, theta, noise) -> None:
    errors = []
    for seed in range(3):
        pts, p1, p2 = sessile_case(bond, theta, noise, seed=seed, tilt_deg=4.0)
        fit = A.fit_arc_spline(pts, p1, p2)
        errors += [fit.theta_p1_deg - theta, fit.theta_p2_deg - theta]
    assert np.sqrt(np.mean(np.square(errors))) < 1.0


def asymmetric_drop(theta_left: float, theta_right: float, bond: float = 1.0, height: float = 150.0):
    """Two Young-Laplace halves of equal apex height and different contact angles."""
    halves = []
    for theta in (theta_left, theta_right):
        h = half_profile(bond, theta, 100.0)
        h = h * (height / h[-1, 1])
        steps = np.hypot(*np.diff(h, axis=0).T)
        s = np.concatenate([[0.0], np.cumsum(steps)])
        grid = np.append(np.arange(0.0, s[-1], 1.0), s[-1])  # keep the contact point
        halves.append(np.column_stack([np.interp(grid, s, h[:, 0]), np.interp(grid, s, h[:, 1])]))
    left, right = halves
    pts = np.vstack([left[::-1] * [-1.0, 1.0], right[1:]])
    pts = pts - [0.0, height]  # contacts on y = 0, apex above
    pts = pts + [320.0, 400.0]
    return pts, pts[0].copy(), pts[-1].copy()


def test_asymmetric_drop_keeps_independent_sides() -> None:
    pts, p1, p2 = asymmetric_drop(70.0, 115.0)
    fit = A.fit_arc_spline(pts, p1, p2)
    assert fit.theta_p1_deg == pytest.approx(70.0, abs=0.3)
    assert fit.theta_p2_deg == pytest.approx(115.0, abs=0.3)


def test_curved_substrate_tangents_change_the_reference() -> None:
    pts, p1, p2 = sessile_case(1.0, 90.0)
    tilt = np.radians(10.0)
    # substrate rising towards the drop at P2: the local tangent turns by 10 deg
    tangents = (np.array([1.0, 0.0]), np.array([np.cos(tilt), -np.sin(tilt)]))
    flat = A.fit_arc_spline(pts, p1, p2)
    curved = A.fit_arc_spline(pts, p1, p2, substrate_tangents=tangents)
    assert curved.theta_p1_deg == pytest.approx(flat.theta_p1_deg, abs=1e-6)
    assert abs(curved.theta_p2_deg - flat.theta_p2_deg) == pytest.approx(10.0, abs=1e-6)


def test_interface_is_extracted_from_a_closed_silhouette() -> None:
    pts, p1, p2 = sessile_case(1.0, 100.0, tilt_deg=6.0)
    closed = closed_silhouette(pts, p1, p2)
    closed = np.roll(closed, 57, axis=0)  # start anywhere, like a real contour
    interface = A.extract_interface(closed, p1, p2)
    assert np.allclose(interface[0], p1) and np.allclose(interface[-1], p2)
    assert len(interface) == pytest.approx(len(pts), abs=3)
    fit, _ = A.fit_sessile_arc_spline(closed, p1, p2)
    assert fit.theta_p1_deg == pytest.approx(100.0, abs=0.1)


def test_diagnostics_are_json_serializable_and_the_model_spans_the_contacts() -> None:
    pts, p1, p2 = sessile_case(2.0, 90.0, "gauss0.5")
    fit = A.fit_arc_spline(pts, p1, p2)
    json.dumps(fit.to_diagnostics())
    model = fit.sample(1.0)
    np.testing.assert_allclose(model[0], p1, atol=1e-6)
    np.testing.assert_allclose(model[-1], p2, atol=1e-6)
    assert np.median(cKDTree(pts).query(model)[0]) < 0.6


@pytest.mark.parametrize(("bond", "theta", "tilt"), [(0.5, 90.0, 0.0), (1.0, 110.0, 5.0), (2.0, 45.0, -4.0)])
def test_image_refinement_moves_the_edge_onto_the_true_boundary(bond, theta, tilt) -> None:
    exact, p1, p2 = sessile_case(bond, theta, contact_radius_px=120.0, tilt_deg=tilt)
    image = render_image(exact, p1, p2, seed=1)
    fit, interface = A.fit_sessile_arc_spline(mask_contour(image), p1, p2)
    refined, points = A.refine_on_image(image, fit, interface, p1, p2)
    truth = cKDTree(exact)
    assert np.median(truth.query(points)[0]) < np.median(truth.query(interface)[0])
    for angle in (refined.theta_p1_deg, refined.theta_p2_deg):
        assert angle == pytest.approx(theta, abs=1.0)


def test_contacts_inside_the_drop_are_moved_onto_the_edge() -> None:
    """A segmentation inside the drop puts the contacts inside too; the arcs,
    forced through them, would bend sharply unless the contacts are refined."""
    exact, p1, p2 = sessile_case(1.0, 110.0, contact_radius_px=120.0)
    image = render_image(exact, p1, p2, seed=0)
    unit = (p2 - p1) / np.hypot(*(p2 - p1))
    q1, q2 = p1 + 5.0 * unit, p2 - 5.0 * unit
    fit, interface = A.fit_sessile_arc_spline(mask_contour(image), q1, q2)
    stuck, _ = A.refine_on_image(image, fit, interface, q1, q2, refine_contacts=False)
    refined, _ = A.refine_on_image(image, fit, interface, q1, q2)
    c1, c2 = refined.contact_points
    assert np.hypot(*(c1 - p1)) < 1.0 and np.hypot(*(c2 - p2)) < 1.0
    assert abs(refined.theta_p1_deg - 110.0) < 1.5
    assert not stuck.accepted or abs(stuck.theta_p1_deg - 110.0) > 5.0


def test_quality_gate_rejects_wrong_contact_points() -> None:
    """A contact point in the middle of the drop cannot yield a sessile fit."""
    pts, p1, p2 = sessile_case(1.0, 90.0)
    wrong_p2 = pts[len(pts) // 2] + np.array([0.0, 300.0])  # under the apex, on the baseline
    closed = closed_silhouette(pts, p1, p2)
    fit, _ = A.fit_sessile_arc_spline(closed, p1, wrong_p2)
    assert not fit.accepted
    assert fit.rejection_reasons


def test_arc_spline_beats_the_linear_tangent_on_rendered_caps() -> None:
    exact, p1, p2 = sessile_case(0.5, 90.0, contact_radius_px=120.0)
    image = render_image(exact, p1, p2, seed=2)
    contour = mask_contour(image)
    line = ((p1[0], p1[1]), (p2[0], p2[1]))
    tangent = tangent_angle_at_point(contour, p1, line, 30, 2.0)[0]
    fit, _ = A.fit_sessile_arc_spline(contour, p1, p2)
    assert abs(fit.theta_p1_deg - 90.0) < abs(tangent - 90.0)


def test_metrics_report_arc_spline_angles_and_diagnostics() -> None:
    exact, p1, p2 = sessile_case(1.0, 110.0, contact_radius_px=120.0, tilt_deg=5.0)
    image = render_image(exact, p1, p2, seed=0)
    contour = mask_contour(image)
    line = ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
    metrics = compute_sessile_metrics(
        contour, px_per_mm=100.0, substrate_line=line,
        contact_points=(tuple(p1), tuple(p2)), contact_angle_method="arc_spline",
        auto_detect_apex=True, image=image,
    )
    assert metrics["method"] == "arc_spline"
    assert metrics["theta_left_deg"] == pytest.approx(110.0, abs=1.0)
    assert metrics["theta_right_deg"] == pytest.approx(110.0, abs=1.0)
    assert metrics["arc_spline"]["accepted"] and metrics["arc_spline"]["image_refined"]
    json.dumps({"a": metrics["arc_spline"], "m": metrics["arc_spline_model_contour_xy"]})


def test_metrics_reject_an_unusable_interface_without_raising() -> None:
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float)
    metrics = compute_sessile_metrics(
        square, px_per_mm=1.0, substrate_line=((0.0, 1.0), (1.0, 1.0)),
        contact_points=((0, 1), (1, 1)), contact_angle_method="arc_spline", apex=(0.5, 0.0),
    )
    assert metrics["arc_spline"]["accepted"] is False
    assert np.isnan(metrics["theta_left_deg"])
    assert metrics["method_left"] == metrics["method_right"] == "rejected"
    gate = metrics["experimental_geometry"]["sessile_arc_spline"]
    assert gate["accepted"] is False and gate["rejection_reasons"]


def test_pipeline_only_refines_on_an_image_that_contains_the_contour() -> None:
    from types import SimpleNamespace

    from menipy.pipelines.sessile.stages import _image_for_refinement

    image = np.zeros((100, 200), np.uint8)
    inside = np.array([[10.0, 10.0], [190.0, 90.0]])
    outside = np.array([[10.0, 10.0], [250.0, 90.0]])
    ctx = SimpleNamespace(image=image, frames=None, image_path=None)
    assert _image_for_refinement(ctx, inside) is image
    assert _image_for_refinement(ctx, outside) is None
