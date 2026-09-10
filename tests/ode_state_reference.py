"""Frozen pre-indexing ODE implementations for exact comparisons and timing."""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from menipy.common.cancellation import check_cancelled


def young_laplace_ode(
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

    apex_curvature = 1.0 / R0_mm
    twice_curvature = 2.0 / R0_mm
    gravity_coefficient = beta / (R0_mm**2)

    def odesys(s, y):
        check_cancelled()
        # y = [r, z, psi]
        r, z, psi = y
        sin_psi = np.sin(psi)

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
            sin_psi_r = apex_curvature
        else:
            sin_psi_r = sin_psi / r

        drds = np.cos(psi)
        dzds = sin_psi
        dpsids = twice_curvature + gravity_coefficient * z - sin_psi_r

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


def sessile_young_laplace_ode(
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

    apex_curvature = 1.0 / R0_mm
    twice_curvature = 2.0 / R0_mm
    gravity_coefficient = Bo / (R0_mm**2)

    def odesys(s, y):
        check_cancelled()
        r, z, psi = y
        sin_psi = np.sin(psi)
        if r < 1e-12:
            sin_psi_r = apex_curvature
        else:
            sin_psi_r = sin_psi / r

        drds = np.cos(psi)
        dzds = sin_psi
        dpsids = twice_curvature + gravity_coefficient * z - sin_psi_r
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


from menipy.common.registry import SOLVERS

SOLVERS.register("young_laplace_ode", young_laplace_ode)
SOLVERS.register("sessile_young_laplace_ode", sessile_young_laplace_ode)

def integrate_young_laplace_profile_mm(
    r0_mm: float,
    beta: float,
    *,
    target_height_mm: float | None = None,
    needle_radius_mm: float | None = None,
    max_step: float = 0.02,
    branch: str = "full",
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Integrate a symmetric pendant Young-Laplace profile in millimetres."""
    r0_mm = float(r0_mm)
    beta = float(beta)
    if not np.isfinite(r0_mm) or not np.isfinite(beta) or r0_mm <= 0:
        profile = np.empty((0, 2), dtype=float)
        meta = {"stop_reason": "invalid_parameters"}
        return (profile, meta) if return_metadata else profile

    z_target = None
    if target_height_mm is not None and target_height_mm > 0:
        z_target = max(float(target_height_mm) / r0_mm, 0.2)

    r_needle_target = None
    if needle_radius_mm is not None and needle_radius_mm > 0:
        r_needle_target = float(needle_radius_mm) / r0_mm

    def ode(_s: float, y: np.ndarray) -> list[float]:
        check_cancelled()
        r, z, psi = y
        sin_psi = np.sin(psi)
        if abs(r) < 1e-10:
            sin_psi_over_r = 1.0
        else:
            sin_psi_over_r = float(sin_psi / r)
        return [
            float(np.cos(psi)),
            float(sin_psi),
            float(2.0 - beta * z - sin_psi_over_r),
        ]

    def hit_axis(s: float, y: np.ndarray) -> float:
        if s <= 0.1:
            return 1.0
        return float(y[0] - 1e-6)

    hit_axis.terminal = True
    hit_axis.direction = -1

    events = [hit_axis]
    if z_target is not None:

        def hit_target_height(_s: float, y: np.ndarray) -> float:
            return float(y[1] - z_target)

        hit_target_height.terminal = True
        hit_target_height.direction = 1
        events.append(hit_target_height)

    if r_needle_target is not None:

        def hit_needle_radius_after_equator(_s: float, y: np.ndarray) -> float:
            r, _z, psi = y
            if psi <= (np.pi / 2.0):
                return 1.0
            return float(r - r_needle_target)

        hit_needle_radius_after_equator.terminal = True
        hit_needle_radius_after_equator.direction = -1
        events.append(hit_needle_radius_after_equator)

    s_max = max(8.0, (z_target or 4.0) * 3.0 + 2.0)
    sol = solve_ivp(
        ode,
        (0.0, s_max),
        [0.0, 0.0, 0.0],
        method="RK45",
        events=events,
        max_step=max_step,
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success or sol.y.shape[1] < 3:
        profile = np.empty((0, 2), dtype=float)
        meta = {"stop_reason": "solver_failed", "solver_message": str(sol.message)}
        return (profile, meta) if return_metadata else profile

    stop_reason = "s_max"
    if sol.t_events:
        event_names = ["axis_return"]
        if z_target is not None:
            event_names.append("height_cutoff")
        if r_needle_target is not None:
            event_names.append("needle_radius")
        for name, events_for_name in zip(event_names, sol.t_events):
            check_cancelled()
            if len(events_for_name) > 0:
                stop_reason = name
                break

    r_right = sol.y[0] * r0_mm
    z_right = sol.y[1] * r0_mm
    if branch == "right":
        profile = np.column_stack([r_right, z_right])
        meta = {"stop_reason": stop_reason, "solver_message": str(sol.message)}
        return (profile, meta) if return_metadata else profile

    r_left = -r_right[::-1]
    z_left = z_right[::-1]
    r_full = np.concatenate([r_left[:-1], r_right])
    z_full = np.concatenate([z_left[:-1], z_right])
    profile = np.column_stack([r_full, z_full])
    meta = {"stop_reason": stop_reason, "solver_message": str(sol.message)}
    return (profile, meta) if return_metadata else profile



