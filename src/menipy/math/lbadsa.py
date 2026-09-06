"""Low-Bond Axisymmetric Drop Shape Analysis (LB-ADSA) mathematical model.

Mathematical formulation based on the analytical first-order perturbation
solution of the Young-Laplace equation for sessile droplets under gravity:
    Stalder, A. F., Melchior, T., Müller, M., Sage, D., Blu, T., & Unser, M. (2010).
    "Low-bond axisymmetric drop shape analysis for surface tension and contact
    angle measurements of sessile drops."
    Colloids and Surfaces A: Physicochemical and Engineering Aspects, 364(1-3), 72-81.
    DOI: 10.1016/j.colsurfa.2010.04.040

Attribution and License Notice:
    The mathematical equations published in Stalder et al. (2010) are public
    scientific theory. This module is an independent, clean-room Python/NumPy
    implementation authored for Menipy. It does not bundle or redistribute any
    third-party source code, bytecode, or binary assets from EPFL.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq


def radius_lbadsa(
    alpha: float | np.ndarray,
    R0: float,
    Bo: float,
) -> float | np.ndarray:
    """Compute the drop profile radius R(alpha) from apex center of curvature.

    In the Stalder (2010) formulation, polar coordinates are centered at the
    apex center of curvature (0, R0). The distance from this center to the
    drop boundary at polar angle alpha (0 <= alpha < pi) is:

        R(alpha) = R0 * (1 + (1/3) * Bo * f(cos(alpha)))

    where:
        f(u) = u * (1/2 + ln(2 / (1 + u))) - 1/2
        u = cos(alpha)

    Args:
        alpha: Polar angle(s) in radians measured from the downward apex normal.
        R0: Apex radius of curvature (in same units as desired output, e.g. px or mm).
        Bo: Dimensionless Bond number (Bo = Delta_rho * g * R0^2 / gamma).

    Returns:
        Radius R(alpha) in the same units as R0.
    """
    alpha_arr = np.asarray(alpha, dtype=float)
    u = np.cos(alpha_arr)
    # Clip cos(alpha) safely away from -1 to avoid log(0) near alpha -> pi
    u_safe = np.clip(u, -0.999999, 1.0)
    log_term = np.log(2.0 / (1.0 + u_safe))
    f_u = u_safe * (0.5 + log_term) - 0.5
    result = R0 * (1.0 + (1.0 / 3.0) * Bo * f_u)

    if np.isscalar(alpha):
        return float(result)
    return result


def dradius_lbadsa(
    alpha: float | np.ndarray,
    R0: float,
    Bo: float,
) -> float | np.ndarray:
    """Compute the derivative dR/dalpha of the LB-ADSA profile radius.

    Analytically differentiated from radius_lbadsa:
        dR/dalpha = (1/3) * Bo * R0 * sin(alpha) * [ u / (1 + u) + ln(1 + u) - 1/2 - ln(2) ]
    where u = cos(alpha).

    At alpha = 0 (apex), dR/dalpha = 0 identically.

    Args:
        alpha: Polar angle(s) in radians.
        R0: Apex radius of curvature.
        Bo: Dimensionless Bond number.

    Returns:
        dR/dalpha in same units as R0 (per radian).
    """
    alpha_arr = np.asarray(alpha, dtype=float)
    u = np.cos(alpha_arr)
    s = np.sin(alpha_arr)
    u_safe = np.clip(u, -0.999999, 1.0)
    # df/dalpha = sin(alpha) * (u / (1 + u) + ln(1 + u) - 0.5 - ln(2))
    df_dalpha = s * (u_safe / (1.0 + u_safe) + np.log(1.0 + u_safe) - 0.5 - np.log(2.0))
    result = (1.0 / 3.0) * Bo * R0 * df_dalpha

    if np.isscalar(alpha):
        return float(result)
    return result


def cartesian_lbadsa(
    alpha: float | np.ndarray,
    R0: float,
    Bo: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Cartesian profile coordinates (X, Z) in the drop apex frame.

    Origin is at the apex: (X=0, Z=0).
    Z points downwards along the symmetry axis into the drop towards substrate.
    X points horizontally outward perpendicular to the symmetry axis.

        X(alpha) = R(alpha) * sin(alpha)
        Z(alpha) = R0 - R(alpha) * cos(alpha)

    Args:
        alpha: Polar angle(s) in radians.
        R0: Apex radius of curvature.
        Bo: Dimensionless Bond number.

    Returns:
        (X, Z) tuple of numpy arrays or floats.
    """
    alpha_arr = np.asarray(alpha, dtype=float)
    r = radius_lbadsa(alpha_arr, R0, Bo)
    x = r * np.sin(alpha_arr)
    z = R0 - r * np.cos(alpha_arr)
    return x, z


def tangent_angle_lbadsa(
    alpha: float | np.ndarray,
    R0: float,
    Bo: float,
) -> float | np.ndarray:
    """Compute the profile tangent angle theta(alpha) in radians.

    The tangent vector t = (dX/dalpha, dZ/dalpha) has components:
        dX/dalpha = R(alpha) * cos(alpha) + dR/dalpha * sin(alpha)
        dZ/dalpha = R(alpha) * sin(alpha) - dR/dalpha * cos(alpha)
        theta(alpha) = atan2(dZ/dalpha, dX/dalpha)

    For Bo = 0 (spherical cap), theta(alpha) = alpha identically.

    Args:
        alpha: Polar angle(s) in radians.
        R0: Apex radius of curvature.
        Bo: Dimensionless Bond number.

    Returns:
        Tangent angle(s) in radians with respect to horizontal.
    """
    alpha_arr = np.asarray(alpha, dtype=float)
    r = radius_lbadsa(alpha_arr, R0, Bo)
    dr = dradius_lbadsa(alpha_arr, R0, Bo)
    dx = r * np.cos(alpha_arr) + dr * np.sin(alpha_arr)
    dz = r * np.sin(alpha_arr) - dr * np.cos(alpha_arr)
    theta = np.arctan2(dz, dx)

    if np.isscalar(alpha):
        return float(theta)
    return theta


def contact_angle_lbadsa(
    R0: float,
    Bo: float,
    height: float,
) -> tuple[float, float]:
    """Find the contact angle theta and contact polar angle alpha_c at height H.

    Solves Z(alpha_c) = H for alpha_c, then evaluates theta(alpha_c).

    Args:
        R0: Apex radius of curvature (same units as height).
        Bo: Dimensionless Bond number.
        height: Vertical distance from drop apex to substrate baseline (H > 0).

    Returns:
        (theta_deg, alpha_c_rad) tuple:
            theta_deg: Contact angle in degrees [0, 180].
            alpha_c_rad: Polar angle at substrate intersection in radians.
    """
    if height <= 0.0 or R0 <= 0.0:
        return 0.0, 0.0

    # For a sphere, cos(alpha_0) = 1 - H / R0
    ratio = np.clip(1.0 - height / R0, -0.9999, 0.9999)
    alpha_init = float(np.arccos(ratio))

    def objective(a: float) -> float:
        r = radius_lbadsa(a, R0, Bo)
        z = R0 - r * np.cos(a)
        return float(z - height)

    # Bracket search around initial guess
    bracket_min = max(1e-4, alpha_init - 0.4)
    bracket_max = min(np.pi - 1e-4, alpha_init + 0.4)

    # If bracket doesn't cross zero, expand safely
    if objective(bracket_min) * objective(bracket_max) > 0:
        bracket_min = 1e-4
        bracket_max = np.pi - 1e-4

    try:
        alpha_c = brentq(objective, bracket_min, bracket_max, xtol=1e-7, rtol=1e-6)
    except (ValueError, RuntimeError):
        # Fallback to spherical cap estimation if root finding fails
        alpha_c = alpha_init

    theta_rad = float(tangent_angle_lbadsa(alpha_c, R0, Bo))
    theta_deg = float(np.degrees(theta_rad))
    return theta_deg, float(alpha_c)


def surface_tension_from_bo(
    Bo: float,
    R0_mm: float,
    delta_rho: float = 998.2,
    g: float = 9.80665,
) -> float:
    """Calculate surface tension gamma in mN/m from Bond number and apex curvature.

        gamma [N/m] = (Delta_rho * g * (R0_mm * 1e-3)^2) / Bo
        gamma [mN/m] = (Delta_rho * g * R0_mm^2 * 1e-3) / Bo

    Args:
        Bo: Dimensionless Bond number (must be positive and non-zero).
        R0_mm: Apex radius of curvature in millimeters.
        delta_rho: Density difference between drop and continuous fluid in kg/m^3.
        g: Gravitational acceleration in m/s^2.

    Returns:
        Surface tension in mN/m, or NaN if Bo is non-physical (<= 0).
    """
    if Bo <= 1e-6 or R0_mm <= 0.0:
        return float("nan")

    gamma_mN_m = (delta_rho * g * (R0_mm**2) * 1e-3) / Bo
    return float(gamma_mN_m)


def lbadsa_ode_profile(
    params: np.ndarray,
    physics: dict[str, Any],
    geometry: dict[str, Any] | None = None,
) -> np.ndarray:
    """Generate a symmetric sessile drop profile using LB-ADSA perturbation.

    Conforms to the Menipy solver signature (integrator for common_solver.run).

    Args:
        params: Array of [R0_mm, Bo].
            R0_mm: Apex radius of curvature in mm.
            Bo: Dimensionless Bond number.
        physics: Physical properties dictionary (contains 'g', 'rho1', 'rho2', etc.).
        geometry: Optional geometry dictionary.

    Returns:
        (N, 2) array of [r, z] coordinates in mm, centered symmetrically with apex at (0, 0).
    """
    if len(params) < 2:
        R0_mm = float(params[0])
        Bo = 0.0
    else:
        R0_mm = float(params[0])
        Bo = float(params[1])

    if R0_mm <= 0.0:
        return np.array([[0.0, 0.0]])

    # Bound Bo to physical perturbation range
    Bo_clamped = float(np.clip(Bo, -0.2, 0.5))

    # Dense sampling of profile up to alpha_max
    alpha_max = np.radians(150.0)
    alphas = np.linspace(0.0, alpha_max, 250)

    x_right, z_right = cartesian_lbadsa(alphas, R0_mm, Bo_clamped)

    # Assemble full symmetric silhouette [r, z]
    r_left = -x_right[::-1]
    z_left = z_right[::-1]

    r_full = np.concatenate([r_left[:-1], x_right])
    z_full = np.concatenate([z_left[:-1], z_right])

    return np.column_stack([r_full, z_full])


# Register with Menipy SOLVERS registry
from menipy.common.registry import SOLVERS

SOLVERS.register("lbadsa", lbadsa_ode_profile)
