"""Two-zone clothoid spline for pendant drops: surface tension, contour, needle angle."""

from __future__ import annotations

import numpy as np
import pytest

from menipy.common import pendant_spline as P
from menipy.common.registry import PENDANT_APPROXIMATORS
from menipy.math.pendant_box import fit_pendant_box
from menipy.models.context import Context
from menipy.models.geometry import Contour
from menipy.pipelines.pendant.stages import (
    PendantPipeline,
    _clip_contour_at_pendant_contacts,
)
from tests.synthetic_pendant import (
    pendant_case,
    pendant_interface,
    pendant_truth,
    render_pendant,
)
from tests.synthetic_sessile import mask_contour

PX_PER_MM = 100.0
PHYSICS = {"rho1": 1000.0, "rho2": 1.2, "g": 9.80665}
DRHO = PHYSICS["rho1"] - PHYSICS["rho2"]
KW = {"px_per_mm": PX_PER_MM, "delta_rho": DRHO, "g": PHYSICS["g"]}


def true_gamma(bond: float) -> float:
    b_m = pendant_truth(bond)["apex_radius_px"] / PX_PER_MM / 1000.0
    return DRHO * PHYSICS["g"] * b_m**2 / bond * 1000.0


def test_zone_layout_indices_are_unique_and_the_equator_is_vertical() -> None:
    lay = P._ZoneLayout((2, 3), (1, 2))
    used = [0, 1]
    for side in (1, 2):
        th = lay.theta_index(side)
        used += list(lay.t_index(side)) + list(lay.h_index(side)) + list(th[th >= 0])
        eq = lay.equator(side)
        assert th[eq] == -1
        assert lay.fixed_theta(side)[eq] == pytest.approx(np.pi / 2)
    used += [lay.contact_index(1), lay.contact_index(2)]
    assert sorted(used) == list(range(lay.size))


def test_analytic_jacobian_with_fixed_equator_heading() -> None:
    pts, p1, p2 = pendant_case(0.3, "gauss0.5", seed=2)
    frame = P._Frame.from_contacts(p1, p2, pts)
    edge = P._Edge(frame.to_local(pts))
    lp1, lp2 = frame.to_local(p1)[0], frame.to_local(p2)[0]
    anchors = P._anchors(edge.q)
    rz = np.column_stack([np.abs(edge.q[:, 0] - anchors["axis"]), edge.q[:, 1] - anchors["apex_y"]])
    shape = fit_pendant_box(rz, anchors["r_eq"], 0.5 * (lp1[1] + lp2[1]) - anchors["apex_y"])
    lay = P._ZoneLayout((2, 2), (1, 2))
    params = P._initial_params(edge, anchors, shape, lay)
    params = params + np.random.default_rng(0).normal(0.0, 0.01, len(params))
    params[0] += 0.3
    problem = P._problem(lay, edge, lp1, lp2, 1.0)
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


@pytest.mark.parametrize("bond", [0.1, 0.25, 0.45])
def test_anchored_box_recovers_bond_and_apex_radius(bond: float) -> None:
    local = pendant_interface(bond)
    truth = pendant_truth(bond)
    shape = fit_pendant_box(np.column_stack([np.abs(local[:, 0]), local[:, 1]]), 150.0, truth["height_px"])
    assert shape.bond == pytest.approx(bond, abs=5e-4)
    assert shape.apex_radius_px == pytest.approx(truth["apex_radius_px"], rel=1e-3)
    assert shape.needle_angle_deg == pytest.approx(truth["needle_angle_deg"], abs=0.05)


@pytest.mark.parametrize("bond", [0.15, 0.3, 0.4])
def test_clean_profiles_give_surface_tension_needle_angle_and_laplace_slope(bond: float) -> None:
    fit = P.fit_pendant_spline(*pendant_case(bond), **KW)
    truth = pendant_truth(bond)
    assert fit.accepted
    assert fit.surface_tension_mN_m == pytest.approx(true_gamma(bond), rel=5e-3)
    assert fit.laplace["surface_tension_mN_m"] == pytest.approx(true_gamma(bond), rel=5e-3)
    assert fit.needle_angle_p1_deg == pytest.approx(truth["needle_angle_deg"], abs=0.1)
    assert fit.needle_angle_p2_deg == pytest.approx(truth["needle_angle_deg"], abs=0.1)
    assert max(sum(fit.zones_p1), sum(fit.zones_p2)) <= 4  # two clothoids per zone at most
    assert fit.rmse_px < 0.1


def test_noisy_edges_and_covariance_uncertainty() -> None:
    gamma_err, needle_err, sigmas = [], [], []
    for seed in range(4):
        fit = P.fit_pendant_spline(*pendant_case(0.3, "gauss1.0", seed=seed), **KW)
        assert fit.accepted
        gamma_err.append(fit.surface_tension_mN_m / true_gamma(0.3) - 1.0)
        needle_err.append(fit.needle_angle_p2_deg - pendant_truth(0.3)["needle_angle_deg"])
        sigmas.append(fit.sigma_p2_deg)
    assert np.sqrt(np.mean(np.square(gamma_err))) < 0.03
    assert np.sqrt(np.mean(np.square(needle_err))) < 1.5
    assert 0.05 < np.mean(sigmas) < 2.0


def test_tilted_scene_keeps_the_axis_and_the_surface_tension() -> None:
    pts, p1, p2 = pendant_case(0.3, "gauss0.5", seed=1, tilt_deg=6.0)
    fit = P.fit_pendant_spline(pts, p1, p2, **KW)
    _, up = fit.axis_image
    expected = np.array([np.sin(np.radians(6.0)), -np.cos(np.radians(6.0))])
    assert np.degrees(np.arccos(np.clip(up @ expected, -1, 1))) < 0.5
    assert fit.surface_tension_mN_m == pytest.approx(true_gamma(0.3), rel=0.02)


def test_contact_level_error_is_absorbed_by_the_symmetry_axis() -> None:
    pts, p1, p2 = pendant_case(0.3, "gauss0.5", seed=1)
    p2_high = p2 + np.array([0.0, -4.0])  # contact detected 4 px up the needle wall
    pts = np.vstack([pts, p2_high])
    fit = P.fit_pendant_spline(pts, p1, p2_high, **KW)
    _, up = fit.axis_image
    assert np.degrees(np.arccos(np.clip(up @ np.array([0.0, -1.0]), -1, 1))) < 0.5
    assert fit.surface_tension_mN_m == pytest.approx(true_gamma(0.3), rel=0.02)


def test_interface_extraction_from_a_pipeline_clipped_mask_contour() -> None:
    image, p1, p2 = render_pendant(0.3, seed=0)
    contour = _clip_contour_at_pendant_contacts(mask_contour(image), np.array([p1, p2]))
    interface = P.extract_pendant_interface(contour, p1, p2)
    np.testing.assert_allclose(interface[0], p1)
    np.testing.assert_allclose(interface[-1], p2)
    steps = np.hypot(*np.diff(interface, axis=0).T)
    assert steps.max() < 3.0
    assert np.all(interface[:, 1] >= min(p1[1], p2[1]) - 1.0)  # no needle shaft


def test_image_refinement_removes_the_mask_contour_bias() -> None:
    image, p1, p2 = render_pendant(0.2, seed=1)
    contour = _clip_contour_at_pendant_contacts(mask_contour(image), np.array([p1, p2]))
    fit, _ = P.fit_pendant_contour(contour, p1, p2, **KW)
    refined, points = P.refine_pendant_on_image(image, fit, p1, p2, **KW)
    gamma = true_gamma(0.2)
    assert abs(refined.surface_tension_mN_m / gamma - 1.0) < 0.01
    assert abs(refined.surface_tension_mN_m / gamma - 1.0) < abs(fit.surface_tension_mN_m / gamma - 1.0)
    assert refined.needle_angle_deg == pytest.approx(pendant_truth(0.2)["needle_angle_deg"], abs=1.0)
    assert len(points) > 200


def _rendered_ctx(model: str, bond: float = 0.3) -> Context:
    image, p1, p2 = render_pendant(bond, seed=2)
    contour = _clip_contour_at_pendant_contacts(mask_contour(image), np.array([p1, p2]))
    ctx = Context(
        frames=[image],
        contour=Contour(xy=contour),
        contact_points=(tuple(int(round(v)) for v in p1), tuple(int(round(v)) for v in p2)),
        scale={"px_per_mm": PX_PER_MM},
        physics=dict(PHYSICS),
        pendant_contour_model=model,
        pendant_approximation_methods=["clothoid_zones"],
    )
    pipe = PendantPipeline()
    pipe.do_geometric_features(ctx)
    pipe.do_profile_fitting(ctx)
    pipe.do_compute_metrics(ctx)
    pipe.do_overlay(ctx)
    return ctx


def test_pipeline_clothoid_zone_contour_model_seeds_the_strict_fit() -> None:
    raw = _rendered_ctx("raw")
    zones = _rendered_ctx("clothoid_zones")
    r = zones.results
    assert r["contour_model"] == "clothoid_zones"
    assert r["strict_fit_success"] is True
    assert r["surface_tension_method"] == "young_laplace_strict"
    assert zones.fit["solver"]["iterations"] < raw.fit["solver"]["iterations"]
    assert r["surface_tension_mN_m"] == pytest.approx(true_gamma(0.3), rel=0.01)
    assert r["clothoid_zones"]["accepted"] is True
    assert r["clothoid_zones"]["image_refined"] is True
    assert r["experimental_geometry"]["pendant_clothoid_zones"]["accepted"] is True
    assert r["needle_angle_deg"] == pytest.approx(pendant_truth(0.3)["needle_angle_deg"], abs=1.0)
    assert r["approx_clothoid_zones_status"] == "ok"
    assert r["approx_clothoid_zones_surface_tension_mN_m"] == r["clothoid_zones_surface_tension_mN_m"]
    assert any(c.get("tag") == "pendant_clothoid_zones" for c in zones.overlay_commands)
    assert "clothoid_zones" not in raw.results
    assert raw.results["approx_clothoid_zones_status"] == "ok"  # the approximator fits by itself


def test_missing_contacts_fall_back_to_the_raw_contour() -> None:
    pts, _, _ = pendant_case(0.3)
    ctx = Context(contour=Contour(xy=pts), scale={"px_per_mm": PX_PER_MM}, physics=dict(PHYSICS),
                  pendant_contour_model="clothoid_zones", pendant_approximation_methods=[])
    pipe = PendantPipeline()
    pipe.do_geometric_features(ctx)
    pipe.do_profile_fitting(ctx)
    pipe.do_compute_metrics(ctx)
    assert ctx.results["contour_model"] == "raw"
    assert ctx.results["clothoid_zones"]["rejection_reasons"] == ["missing_contact_points"]
    assert ctx.results["strict_fit_success"] is True


def test_clothoid_zones_approximator_is_registered_but_not_default() -> None:
    from menipy.pipelines.pendant.stages import DEFAULT_PENDANT_APPROXIMATION_METHODS

    assert "clothoid_zones" in PENDANT_APPROXIMATORS
    assert "clothoid_zones" not in DEFAULT_PENDANT_APPROXIMATION_METHODS
