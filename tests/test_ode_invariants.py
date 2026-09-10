"""Reference integrations for invariant callback calculations."""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from menipy.common.cancellation import check_cancelled


def legacy_pendant(
    params: np.ndarray,
    physics: dict[str, Any],
    geometry: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Integrate the axisymmetric Young-Laplace ODE using Bashforth-Adams formulation.

    Args:
        params: Array of [R0_mm, beta]
            R0_mm: Radius of curvature at the apex in mm.
            beta: Bond number (dimensionless shape parameter).
        physics: Dictionary containing physical properties.
        geometry: Optional geometry dictionary (ignored, included for signature compatibility).

    Returns:
        (N, 2) array of [r, z] profile coordinates in mm.
    """
    if len(params) < 2:
        # Fallback if only R0 is provided
        R0_mm = float(params[0])
        beta = 0.0
    else:
        R0_mm = float(params[0])
        beta = float(params[1])

    # For safety against extreme params
    if R0_mm <= 0:
        return np.array([[0.0, 0.0]])

    def odesys(s, y):
        check_cancelled()
        # y = [r, z, psi]
        r, z, psi = y

        # Avoid division by zero at apex (s=0, r=0)
        # Using L'Hopital's rule: lim(s->0) sin(psi)/r = dpsi/ds
        # Since r=0, z=0, dpsi/ds = 2/R0_mm + beta*z = 2/R0_mm
        # And since dimensionless, if we scale by R0 it is 2.
        # We integrate in real units (mm), but usually beta is defined dimensionless.
        # The ODE in real units (arc-length s in mm):
        # dr/ds = cos(psi)
        # dz/ds = sin(psi)
        # dpsi/ds = 2/R0_mm + (beta/R0_mm^2) * z - sin(psi)/r

        if r < 1e-12:
            sin_psi_r = 1.0 / R0_mm
        else:
            sin_psi_r = np.sin(psi) / r

        drds = np.cos(psi)
        dzds = np.sin(psi)
        dpsids = (2.0 / R0_mm) + (beta / (R0_mm**2)) * z - sin_psi_r

        return [drds, dzds, dpsids]

    # Stop integration when profile curls back up to axis (pendant drop pinch-off)
    # or max length reached
    def hit_axis(s, y):
        return y[0] - 1e-6 if s > 0.1 else 1.0

    hit_axis.terminal = True
    hit_axis.direction = -1

    # Max arc length to integrate (scale by R0)
    s_max = 5.0 * R0_mm * max(1.0, 1.0 / max(1e-3, abs(beta)))

    y0 = [0.0, 0.0, 0.0]

    # We want a dense enough output to compare with contours
    sol = solve_ivp(
        odesys,
        [0.0, s_max],
        y0,
        method="RK45",
        events=hit_axis,
        max_step=s_max / 200.0,
        rtol=1e-5,
        atol=1e-6,
    )

    # Combine the symmetric halves (left and right)
    # sol.y[0] is r, sol.y[1] is z
    r_right = sol.y[0]
    z_right = sol.y[1]

    # For a full drop silhouette, we mirror the r coordinate
    # But usually the solver residual function compares against [r, z] format.
    # The solver pointwise expects (x, y) where x is horizontal and y is vertical.
    # We center r around 0, appending the left side.
    r_left = -r_right[::-1]
    z_left = z_right[::-1]

    r_full = np.concatenate([r_left[:-1], r_right])
    z_full = np.concatenate([z_left[:-1], z_right])

    return np.column_stack([r_full, z_full])


def legacy_sessile(
    params: np.ndarray,
    physics: dict[str, Any],
    geometry: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Integrate the axisymmetric Young-Laplace ODE for a sessile drop.

    Formulation:
        dr/ds = cos(psi)
        dz/ds = sin(psi)
        dpsi/ds = 2/R0 + (Bo / R0^2) * z - sin(psi)/r

    Academic References:
        1. Rotenberg, Y., Boruvka, L., & Neumann, A. W. (1983).
           "Determination of surface tension and contact angle from the shapes of axisymmetric fluid interfaces."
           J. Colloid Interface Sci., 93(1), 169-183. DOI: 10.1016/0021-9797(83)90396-X
        2. Bateni, A., et al. (2003).
           "Axisymmetric drop shape analysis-contact diameter (ADSA-CD)."
           Colloids Surf. A, 219(1-3), 215-231. DOI: 10.1016/S0927-7757(03)00037-7

    Args:
        params: Array of [R0_mm, Bo]
        physics: Dictionary with physical properties.
        geometry: Optional geometry dictionary (may supply 'target_height_mm' or 'height_mm').

    Returns:
        (N, 2) array of [r, z] profile coordinates in mm (full symmetric silhouette).
    """
    if len(params) < 2:
        R0_mm = float(params[0])
        Bo = 0.0
    else:
        R0_mm = float(params[0])
        Bo = float(params[1])

    if R0_mm <= 0:
        return np.array([[0.0, 0.0]])

    target_height_mm = None
    if geometry and "height_mm" in geometry and geometry["height_mm"] is not None:
        try:
            target_height_mm = float(geometry["height_mm"])
        except (TypeError, ValueError):
            target_height_mm = None

    def odesys(s, y):
        check_cancelled()
        r, z, psi = y
        if r < 1e-12:
            sin_psi_r = 1.0 / R0_mm
        else:
            sin_psi_r = np.sin(psi) / r

        drds = np.cos(psi)
        dzds = np.sin(psi)
        dpsids = (2.0 / R0_mm) + (Bo / (R0_mm**2)) * z - sin_psi_r
        return [drds, dzds, dpsids]

    events = []
    if target_height_mm is not None and target_height_mm > 0:

        def hit_target_h(s, y):
            return y[1] - target_height_mm

        hit_target_h.terminal = True
        hit_target_h.direction = 1
        events.append(hit_target_h)

    def hit_overhang(s, y):
        return (np.pi * 175.0 / 180.0) - y[2]

    hit_overhang.terminal = True
    hit_overhang.direction = -1
    events.append(hit_overhang)

    s_max = max(5.0 * R0_mm, (target_height_mm or R0_mm) * 3.5)
    y0 = [0.0, 0.0, 0.0]

    sol = solve_ivp(
        odesys,
        [0.0, s_max],
        y0,
        method="RK45",
        events=events,
        max_step=s_max / 150.0,
        rtol=1e-5,
        atol=1e-6,
    )

    r_right = sol.y[0]
    z_right = sol.y[1]

    r_left = -r_right[::-1]
    z_left = z_right[::-1]

    r_full = np.concatenate([r_left[:-1], r_right])
    z_full = np.concatenate([z_left[:-1], z_right])

    return np.column_stack([r_full, z_full])


import pytest

from menipy.math import young_laplace as yl


@pytest.mark.parametrize(
    "params", [[2.0], [1.2, 0], [1.2, 0.6], [0.1, 3.0], [5.0, -0.1], [-1.0, 0.6]]
)
@pytest.mark.parametrize("height", [None, 0.5, 2.0, 10.0])
def test_exact_adaptive_profiles(params, height):
    for current, reference in (
        (yl.young_laplace_ode, legacy_pendant),
        (yl.sessile_young_laplace_ode, legacy_sessile),
    ):
        geometry = {"height_mm": height}
        actual = current(np.array(params), {}, geometry)
        expected = reference(np.array(params), {}, geometry)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "function", [yl.young_laplace_ode, yl.sessile_young_laplace_ode]
)
def test_callback_cancellation(function, monkeypatch):
    from menipy.common.cancellation import AnalysisCancelled

    calls = []

    def cancel():
        calls.append(True)
        if len(calls) == 3:
            raise AnalysisCancelled()

    monkeypatch.setattr(yl, "check_cancelled", cancel)
    with pytest.raises(AnalysisCancelled):
        function(np.array([1.2, 0.6]), {})
    assert len(calls) == 3


def fitted_output(integrator):
    from types import SimpleNamespace

    from menipy.common import solver
    from menipy.models.fit import FitConfig

    geometry = {"height_mm": 1.8}
    obs = integrator(np.array([2.0, 0.25]), {}, geometry)
    obs = obs + np.random.default_rng(82).normal(0, 0.0001, obs.shape)
    ctx = SimpleNamespace(
        contour=SimpleNamespace(xy=obs, units="mm"), geometry=geometry
    )
    result = solver.run(
        ctx,
        integrator=integrator,
        config=FitConfig(x0=[1.9, 0.2], bounds=([0.5, 0.01], [4.0, 1.0])),
    )
    del result["solver"]["time_ms"]
    return result


@pytest.mark.parametrize(
    "current,reference",
    [
        (yl.young_laplace_ode, legacy_pendant),
        (yl.sessile_young_laplace_ode, legacy_sessile),
    ],
)
def test_exact_fit(current, reference):
    assert fitted_output(current) == fitted_output(reference)
