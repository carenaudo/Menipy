import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, Any, Optional


def young_laplace_ode(
    params: np.ndarray,
    physics: Dict[str, Any],
    geometry: Optional[Dict[str, Any]] = None,
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


from menipy.common.registry import SOLVERS

SOLVERS.register("young_laplace_ode", young_laplace_ode)
