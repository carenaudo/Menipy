"""Profile fitting algorithms for droplets with immersed dispensing needles.

Because a dispensing needle pierces the droplet apex, axisymmetric Young-Laplace
(ADSA) boundary conditions cannot be applied.  This module provides two
specialised methods tailored for needle-in-sessile-drop measurements:

1. **Tangent Method 2**: Local cubic polynomial regression in substrate-aligned
   coordinates near each three-phase contact point.
2. **CDF (Circumcircle and Difference Fitting)**: Circumcircle baseline fitting
   with low-order polynomial residual correction (*Albert et al., ACS Omega
   2019*).

References
----------
Albert, E., Tegze, B., Hajnal, Z., Zámbó, D., Szekrényes, D. P., Deák, A.,
Hórvölgyi, Z., & Nagy, N. (2019). Robust Contact Angle Determination for
Needle-in-Drop Type Measurements. ACS Omega, 4(19), 18465–18471.
DOI: 10.1021/acsomega.9b02857.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NeedleProfileFitResult:
    """Result of needle-in-drop contact angle fitting for a single frame."""

    theta_left_deg: float
    """Contact angle at left contact point in degrees."""
    theta_right_deg: float
    """Contact angle at right contact point in degrees."""
    uncertainty_left: float
    """Fit root-mean-square error (RMSE) for left side in pixels."""
    uncertainty_right: float
    """Fit RMSE for right side in pixels."""
    method: str
    """Method used: 'tangent', 'cdf', or 'auto'."""
    rmse_left_px: float
    """Left profile residual RMSE in pixels."""
    rmse_right_px: float
    """Right profile residual RMSE in pixels."""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    """Diagnostic details including method comparison."""


def _substrate_frame(
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    contact_point: tuple[float, float],
    is_left: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inward tangent and upward normal unit vectors for a contact point.

    Parameters
    ----------
    substrate_line : ((x1, y1), (x2, y2))
        Substrate baseline endpoints.
    contact_point : (x, y)
        Baseline contact point.
    is_left : bool
        True if left contact point (inward points to the right).
    """
    p1 = np.asarray(substrate_line[0], dtype=float)
    p2 = np.asarray(substrate_line[1], dtype=float)
    line_vec = p2 - p1
    length = float(np.linalg.norm(line_vec))
    if length <= 1e-9:
        u = np.array([1.0, 0.0])
    else:
        u = line_vec / length

    # Ensure u points left-to-right (positive x)
    if u[0] < 0:
        u = -u

    # Normal pointing upward (negative image y in OpenCV)
    v = np.array([-u[1], u[0]])
    if v[1] > 0:  # if pointing downward in image coordinates, flip to upward
        v = -v

    # Inward tangent along substrate towards drop center
    inward = u if is_left else -u
    return inward, v


def _extract_side_contour(
    contour: np.ndarray,
    contact_point: tuple[float, float],
    inward_vec: np.ndarray,
    normal_vec: np.ndarray,
    *,
    max_height_px: float = 120.0,
    max_dist_px: float = 150.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and transform lateral contour points near a contact point.

    Returns coordinates (xi, eta) where xi is inward along substrate and
    eta is upward along substrate normal.
    """
    pts = np.asarray(contour, dtype=float).reshape(-1, 2)
    cp = np.asarray(contact_point, dtype=float)
    delta = pts - cp

    # Project onto substrate-aligned frame
    xi = delta @ inward_vec
    eta = delta @ normal_vec

    # Keep points in the upper half-plane (eta >= -1.0) and within distance
    dist = np.hypot(xi, eta)
    mask = (eta >= -1.5) & (eta <= max_height_px) & (dist <= max_dist_px) & (dist >= 0.5)

    xi_sub = xi[mask]
    eta_sub = eta[mask]

    # Sort by increasing arc distance from contact point
    order = np.argsort(dist[mask])
    return xi_sub[order], eta_sub[order]


# ---------------------------------------------------------------------------
# Method A: Tangent Method 2 (Local Polynomial)
# ---------------------------------------------------------------------------


def fit_contact_angle_tangent(
    contour: np.ndarray,
    contact_point: tuple[float, float],
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    is_left: bool,
    *,
    window_px: int = 35,
) -> tuple[float, float]:
    """Fit local polynomial near contact point and return (angle_deg, rmse_px)."""
    inward, normal = _substrate_frame(substrate_line, contact_point, is_left)
    xi, eta = _extract_side_contour(
        contour, contact_point, inward, normal, max_dist_px=float(window_px)
    )

    if len(xi) < 4:
        return 90.0, 10.0

    # Weights decreasing with distance from contact line
    dist = np.hypot(xi, eta)
    weights = 1.0 / np.maximum(dist, 1.0)

    # Determine whether profile is shallow (eta = f(xi)) or steep (xi = f(eta))
    # Check orientation of the furthest point in window
    aspect = float(np.median(np.abs(eta)) / max(1e-3, np.median(np.abs(xi))))

    if aspect > 1.2:
        # Steep angle (theta > ~50°): fit xi = b1*eta + b2*eta^2 + b3*eta^3
        # Substrate contact condition: xi(0) = 0
        deg = 2 if len(xi) < 7 else 3
        design = np.column_stack([eta**k for k in range(1, deg + 1)])
        try:
            w_diag = weights[:, None]
            b, residuals, _, _ = np.linalg.lstsq(design * w_diag, xi * weights, rcond=None)
            # at eta=0: dxi/deta = b[0]
            # Since theta is angle between inward substrate and profile tangent:
            # inward = +xi axis, upward = +eta axis
            # dxi/deta = cot(theta) = 1 / tan(theta)
            # theta = arctan2(1.0, b[0])
            theta_rad = np.arctan2(1.0, float(b[0]))
            if theta_rad < 0:
                theta_rad += np.pi
            angle_deg = float(np.degrees(theta_rad))

            pred_xi = design @ b
            rmse = float(np.sqrt(np.mean((xi - pred_xi) ** 2)))
            return angle_deg, rmse
        except Exception:
            pass

    # Shallow or moderate angle: fit eta = a1*xi + a2*xi^2 + a3*xi^3
    deg = 2 if len(xi) < 7 else 3
    design = np.column_stack([xi**k for k in range(1, deg + 1)])
    try:
        w_diag = weights[:, None]
        a, _, _, _ = np.linalg.lstsq(design * w_diag, eta * weights, rcond=None)
        # at xi=0: deta/dxi = a[0] = tan(theta)
        theta_rad = np.arctan2(float(a[0]), 1.0)
        if theta_rad < 0:
            theta_rad += np.pi
        angle_deg = float(np.degrees(theta_rad))

        pred_eta = design @ a
        rmse = float(np.sqrt(np.mean((eta - pred_eta) ** 2)))
        return angle_deg, rmse
    except Exception:
        return 90.0, 10.0


# ---------------------------------------------------------------------------
# Method B: Circumcircle and Difference Fitting (CDF)
# ---------------------------------------------------------------------------


def fit_contact_angle_cdf(
    contour: np.ndarray,
    contact_point: tuple[float, float],
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    is_left: bool,
    *,
    max_height_px: float = 90.0,
) -> tuple[float, float]:
    """Fit contact angle via Circumcircle and Difference Fitting (Albert et al. 2019).

    Parameters
    ----------
    contour : np.ndarray
        Raw drop contour points.
    contact_point : (x, y)
        Base contact point.
    substrate_line : line tuple
        Detected baseline.
    is_left : bool
        Left or right contact point.
    max_height_px : float
        Maximum height along normal to include (avoids needle zone).

    Returns
    -------
    (angle_deg, rmse_px)
    """
    inward, normal = _substrate_frame(substrate_line, contact_point, is_left)
    xi, eta = _extract_side_contour(
        contour, contact_point, inward, normal, max_height_px=max_height_px, max_dist_px=150.0
    )

    if len(xi) < 8:
        # Fallback to tangent method if too few points for circle
        return fit_contact_angle_tangent(contour, contact_point, substrate_line, is_left)

    # 1. Fit circumcircle in local coordinates (xi, eta) passing near (0, 0)
    # Equation: (xi - c_xi)^2 + (eta - c_eta)^2 = R^2
    # Linear form: 2*c_xi*xi + 2*c_eta*eta + (R^2 - c_xi^2 - c_eta^2) = xi^2 + eta^2
    a_mat = np.column_stack([2.0 * xi, 2.0 * eta, np.ones_like(xi)])
    b_vec = xi**2 + eta**2

    try:
        sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        c_xi, c_eta, k = float(sol[0]), float(sol[1]), float(sol[2])
        r_sq = k + c_xi**2 + c_eta**2
        if r_sq <= 0:
            return fit_contact_angle_tangent(contour, contact_point, substrate_line, is_left)
        radius = np.sqrt(r_sq)

        # Circumcircle analytical tangent at baseline eta = 0:
        # At eta = 0: (xi - c_xi)^2 + c_eta^2 = R^2 => xi_base = c_xi ± sqrt(R^2 - c_eta^2)
        disc = radius**2 - c_eta**2
        if disc < 0:
            return fit_contact_angle_tangent(contour, contact_point, substrate_line, is_left)

        # Baseline intersection closest to (0, 0)
        xi_cand1 = c_xi + np.sqrt(disc)
        xi_cand2 = c_xi - np.sqrt(disc)
        xi_base = xi_cand1 if abs(xi_cand1) < abs(xi_cand2) else xi_cand2

        # Derivative from circle equation at (xi_base, 0):
        # 2*(xi - c_xi) + 2*(eta - c_eta)*deta/dxi = 0
        # deta/dxi = -(xi - c_xi) / (eta - c_eta) = -(xi_base - c_xi) / (-c_eta) = (xi_base - c_xi) / c_eta
        if abs(c_eta) < 1e-6:
            slope_circ = 0.0
        else:
            slope_circ = (xi_base - c_xi) / c_eta

        # 2. Difference calculation: Delta(xi) = eta_observed - eta_circle(xi)
        # eta_circ(xi) = c_eta + sqrt(R^2 - (xi - c_xi)^2)  (upper arc)
        rad_term = radius**2 - (xi - c_xi) ** 2
        valid = rad_term > 0
        if np.count_nonzero(valid) < 6:
            theta_rad = np.arctan2(slope_circ, 1.0)
            if theta_rad < 0:
                theta_rad += np.pi
            return float(np.degrees(theta_rad)), 1.5

        eta_circ = c_eta + np.sqrt(np.maximum(rad_term[valid], 0.0))
        delta = eta[valid] - eta_circ
        xi_val = xi[valid]

        # 3. Fit low-order difference polynomial: Delta(xi) = d1*xi + d2*xi^2
        diff_design = np.column_stack([xi_val, xi_val**2])
        d_coef, _, _, _ = np.linalg.lstsq(diff_design, delta, rcond=None)
        d_delta_dxi_base = float(d_coef[0])

        # Total slope = circle slope + difference derivative at base
        total_slope = slope_circ + d_delta_dxi_base
        theta_rad = np.arctan2(total_slope, 1.0)
        if theta_rad < 0:
            theta_rad += np.pi

        angle_deg = float(np.degrees(theta_rad))
        pred_eta = eta_circ + diff_design @ d_coef
        rmse = float(np.sqrt(np.mean((eta[valid] - pred_eta) ** 2)))
        return angle_deg, rmse

    except Exception:
        return fit_contact_angle_tangent(contour, contact_point, substrate_line, is_left)


# ---------------------------------------------------------------------------
# Unified Fitting Interface
# ---------------------------------------------------------------------------


def fit_needle_contact_angles(
    contour: np.ndarray,
    contact_points: tuple[tuple[float, float], tuple[float, float]],
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    *,
    method: str = "auto",
    needle_rect: tuple[int, int, int, int] | None = None,
) -> NeedleProfileFitResult:
    """Compute left and right contact angles using needle-safe models.

    Parameters
    ----------
    contour : np.ndarray
        Full droplet contour.
    contact_points : ((xL, yL), (xR, yR))
        Left and right baseline contact coordinates.
    substrate_line : ((x1, y1), (x2, y2))
        Baseline line endpoints.
    method : str
        ``"tangent"``, ``"cdf"``, or ``"auto"`` (selects method with lower RMSE).
    needle_rect : (x, y, w, h) | None
        Optional needle bounding box to cap the maximum height search band.

    Returns
    -------
    NeedleProfileFitResult
        Measured contact angles, uncertainties, and method diagnostics.
    """
    p_left, p_right = contact_points

    max_h = 100.0
    if needle_rect is not None:
        # Needle bottom immersion point relative to substrate
        sub_y = (substrate_line[0][1] + substrate_line[1][1]) / 2.0
        needle_bottom_y = needle_rect[1] + needle_rect[3]
        max_h = max(20.0, float(sub_y - needle_bottom_y - 4.0))

    if method == "tangent":
        ang_l, rmse_l = fit_contact_angle_tangent(contour, p_left, substrate_line, True)
        ang_r, rmse_r = fit_contact_angle_tangent(contour, p_right, substrate_line, False)
        return NeedleProfileFitResult(
            theta_left_deg=round(ang_l, 2),
            theta_right_deg=round(ang_r, 2),
            uncertainty_left=round(rmse_l, 3),
            uncertainty_right=round(rmse_r, 3),
            method="tangent",
            rmse_left_px=round(rmse_l, 3),
            rmse_right_px=round(rmse_r, 3),
        )

    if method == "cdf":
        ang_l, rmse_l = fit_contact_angle_cdf(
            contour, p_left, substrate_line, True, max_height_px=max_h
        )
        ang_r, rmse_r = fit_contact_angle_cdf(
            contour, p_right, substrate_line, False, max_height_px=max_h
        )
        return NeedleProfileFitResult(
            theta_left_deg=round(ang_l, 2),
            theta_right_deg=round(ang_r, 2),
            uncertainty_left=round(rmse_l, 3),
            uncertainty_right=round(rmse_r, 3),
            method="cdf",
            rmse_left_px=round(rmse_l, 3),
            rmse_right_px=round(rmse_r, 3),
        )

    # method == "auto": evaluate both, choose lowest RMSE per side
    ang_l_tan, rmse_l_tan = fit_contact_angle_tangent(contour, p_left, substrate_line, True)
    ang_r_tan, rmse_r_tan = fit_contact_angle_tangent(contour, p_right, substrate_line, False)
    ang_l_cdf, rmse_l_cdf = fit_contact_angle_cdf(
        contour, p_left, substrate_line, True, max_height_px=max_h
    )
    ang_r_cdf, rmse_r_cdf = fit_contact_angle_cdf(
        contour, p_right, substrate_line, False, max_height_px=max_h
    )

    # Prefer Tangent unless CDF is materially better (lower RMSE)
    if rmse_l_cdf < rmse_l_tan * 0.90:
        chosen_l = (ang_l_cdf, rmse_l_cdf, "cdf")
    else:
        chosen_l = (ang_l_tan, rmse_l_tan, "tangent")

    if rmse_r_cdf < rmse_r_tan * 0.90:
        chosen_r = (ang_r_cdf, rmse_r_cdf, "cdf")
    else:
        chosen_r = (ang_r_tan, rmse_r_tan, "tangent")

    return NeedleProfileFitResult(
        theta_left_deg=round(chosen_l[0], 2),
        theta_right_deg=round(chosen_r[0], 2),
        uncertainty_left=round(chosen_l[1], 3),
        uncertainty_right=round(chosen_r[1], 3),
        method=f"{chosen_l[2]}/{chosen_r[2]}",
        rmse_left_px=round(chosen_l[1], 3),
        rmse_right_px=round(chosen_r[1], 3),
        diagnostics={
            "left": {"tangent": (ang_l_tan, rmse_l_tan), "cdf": (ang_l_cdf, rmse_l_cdf)},
            "right": {"tangent": (ang_r_tan, rmse_r_tan), "cdf": (ang_r_cdf, rmse_r_cdf)},
        },
    )
