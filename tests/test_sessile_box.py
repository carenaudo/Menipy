"""Box-matched Young-Laplace profiles and the golden-section Bond search."""

from __future__ import annotations

import numpy as np
import pytest

from menipy.math.sessile_box import (
    fit_side_box,
    golden_section_minimize,
    profile_table,
)
from tests.synthetic_sessile import half_profile


def test_zero_bond_row_is_a_circle_of_unit_apex_radius() -> None:
    table = profile_table()
    alive = table.phi[0] < np.pi
    s = table.s[alive]
    np.testing.assert_allclose(table.x[0][alive], np.sin(s), atol=1e-6)
    np.testing.assert_allclose(table.z[0][alive], 1.0 - np.cos(s), atol=1e-6)
    np.testing.assert_allclose(table.kappa[0][alive], 1.0, atol=1e-4)


def test_golden_section_finds_the_minimum_of_a_unimodal_function() -> None:
    x, fx, n = golden_section_minimize(lambda v: (v - 0.3) ** 2 + 1.0, 0.0, 1.0, tol=1e-6)
    assert x == pytest.approx(0.3, abs=1e-5)
    assert fx == pytest.approx(1.0, abs=1e-9)
    # each iteration shrinks the bracket by the golden ratio: ~ log(tol)/log(0.618)
    assert n < 35


@pytest.mark.parametrize(
    ("bond", "theta"),
    [(0.0, 40.0), (0.5, 90.0), (2.0, 120.0), (8.0, 150.0), (20.0, 60.0)],
)
def test_box_match_recovers_bond_number_and_contact_angle(bond: float, theta: float) -> None:
    half = half_profile(bond, theta)
    pts = half[::20] + np.random.default_rng(1).normal(0.0, 0.3, (len(half[::20]), 2))
    shape = fit_side_box(pts, float(half[-1, 0]), float(half[-1, 1]))
    assert shape.theta_deg == pytest.approx(theta, abs=0.3)
    if bond > 0.3:  # near-spherical drops barely constrain Bo
        assert shape.bond == pytest.approx(bond, rel=0.1)
    # the matched profile ends exactly on the box corner
    assert shape.x_px[-1] == pytest.approx(half[-1, 0])
    assert shape.z_px[-1] == pytest.approx(half[-1, 1])


def test_arc_count_follows_curvature_variation_and_tolerance() -> None:
    sphere = fit_side_box(half_profile(0.0, 90.0)[::20], 300.0, 300.0)
    flat = fit_side_box(half_profile(8.0, 150.0)[::20], *half_profile(8.0, 150.0)[-1])
    assert sphere.arcs_needed(0.05) == 1  # constant curvature: one arc per side
    assert flat.arcs_needed(0.05) > flat.arcs_needed(0.5) > 1
    joins = flat.join_positions(5)
    assert np.all(np.diff(joins) > 0)
    assert 0.0 < joins[0] and joins[-1] < flat.s_px[-1]
    # the curvature rises towards the contact line, so the joins crowd there
    assert np.diff(joins)[-1] < np.diff(joins)[0]


def test_degenerate_box_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_side_box(np.zeros((10, 2)), 0.0, 10.0)
