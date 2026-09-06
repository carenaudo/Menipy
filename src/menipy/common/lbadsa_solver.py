"""Low-Bond Axisymmetric Drop Shape Analysis (LB-ADSA) fitting solver.

Fits the first-order perturbation model of the Young-Laplace equation to
extracted sessile drop contours using non-linear least squares.

Academic Reference:
    Stalder, A. F., Melchior, T., Müller, M., Sage, D., Blu, T., & Unser, M. (2010).
    "Low-bond axisymmetric drop shape analysis for surface tension and contact
    angle measurements of sessile drops."
    Colloids and Surfaces A: Physicochemical and Engineering Aspects, 364(1-3), 72-81.
    DOI: 10.1016/j.colsurfa.2010.04.040

Attribution and Clean-Room Notice:
    This solver is an independent clean-room implementation developed for Menipy
    using SciPy Levenberg-Marquardt / Trust Region Reflective optimization with
    direct ray-projection residuals. No proprietary or third-party code from
    EPFL's ImageJ plugins is used or redistributed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from menipy.common.geometry import fit_circle
from menipy.math.lbadsa import (
    cartesian_lbadsa,
    contact_angle_lbadsa,
    radius_lbadsa,
    surface_tension_from_bo,
)
from menipy.models.geometry import SubstrateProfile


def fit_lbadsa_drop(
    contour_xy: np.ndarray,
    substrate_line_or_profile: tuple[tuple[float, float], tuple[float, float]]
    | SubstrateProfile
    | None,
    apex_xy: tuple[float, float] | None = None,
    contact_points: tuple[tuple[float, float], tuple[float, float]] | None = None,
    px_per_mm: float = 1.0,
    physics: dict[str, Any] | None = None,
    optimize_bo: bool = True,
) -> dict[str, Any]:
    """Fit LB-ADSA perturbation model to a sessile drop contour.

    Args:
        contour_xy: (N, 2) array of contour coordinates [x, y] in pixels.
        substrate_line_or_profile: Substrate line ((x1, y1), (x2, y2)) or SubstrateProfile.
        apex_xy: Optional apex coordinate (x, y). If None, estimated from min y.
        contact_points: Optional pre-computed contact points ((xL, yL), (xR, yR)).
        px_per_mm: Calibrated spatial scale in pixels per millimeter.
        physics: Optional physics dictionary containing 'rho1', 'rho2', 'g', etc.
        optimize_bo: If True, optimizes Bond number along with R0 and apex offset.

    Returns:
        Dictionary containing:
            theta_left_deg: Left contact angle in degrees.
            theta_right_deg: Right contact angle in degrees.
            theta_mean_deg: Mean contact angle in degrees.
            bond_number: Fitted or constrained Bond number Bo.
            R0_px: Apex radius of curvature in pixels.
            R0_mm: Apex radius of curvature in millimeters.
            surface_tension_mN_m: Estimated surface tension in mN/m (if physical).
            rmse_px: Root mean square radial fitting residual in pixels.
            max_residual_px: Maximum absolute residual in pixels.
            model_contour_xy: Fitted model curve coordinates in image pixels.
            diagnostics: Detailed solver diagnostic dictionary.
    """
    xy = np.asarray(contour_xy, dtype=float).reshape(-1, 2)
    if len(xy) < 10:
        return _empty_result("insufficient_contour_points")

    # 1. Resolve substrate line and profile
    sub_profile: SubstrateProfile | None = None
    sub_line: tuple[tuple[float, float], tuple[float, float]] | None = None

    if isinstance(substrate_line_or_profile, SubstrateProfile):
        sub_profile = substrate_line_or_profile
        sub_line = sub_profile.to_chord()
    elif substrate_line_or_profile is not None:
        sub_line = substrate_line_or_profile
        sub_profile = SubstrateProfile.from_line(sub_line[0], sub_line[1])
    else:
        # Fallback horizontal substrate at max y
        max_y = float(np.max(xy[:, 1]))
        min_x = float(np.min(xy[:, 0]))
        max_x = float(np.max(xy[:, 0]))
        sub_line = ((min_x, max_y), (max_x, max_y))
        sub_profile = SubstrateProfile.from_line(sub_line[0], sub_line[1])

    # 2. Substrate frame vectors
    p1_sub = np.asarray(sub_line[0], dtype=float)
    p2_sub = np.asarray(sub_line[1], dtype=float)
    s_sub = p2_sub - p1_sub
    s_norm = np.linalg.norm(s_sub)
    if s_norm < 1e-6:
        return _empty_result("invalid_substrate_line")
    s_hat = s_sub / s_norm

    # Upward normal (towards apex)
    n_candidate = np.array([-s_hat[1], s_hat[0]], dtype=float)

    # Resolve apex
    if apex_xy is not None:
        apex_pt = np.asarray(apex_xy, dtype=float)
    else:
        min_idx = int(np.argmin(xy[:, 1]))
        apex_pt = xy[min_idx].copy()

    # Ensure normal points from substrate towards apex
    if np.dot(apex_pt - p1_sub, n_candidate) < 0:
        n_candidate = -n_candidate

    n_hat = n_candidate  # Upward normal into droplet
    z_hat = -n_hat  # Symmetry axis downwards from apex towards substrate
    x_hat = np.array([-z_hat[1], z_hat[0]], dtype=float)  # Horizontal perpendicular

    # 3. Transform contour points into apex frame (X: horizontal, Z: downwards along axis)
    vecs = xy - apex_pt
    Z_all = np.dot(vecs, z_hat)
    X_all = np.dot(vecs, x_hat)

    # Compute droplet height at apex
    height_apex = float(abs(np.dot(p1_sub - apex_pt, n_hat)))
    if height_apex <= 1.0:
        return _empty_result("droplet_height_too_small")

    # Filter points belonging to the droplet above substrate
    valid_mask = (Z_all >= -2.0) & (Z_all <= height_apex + 5.0)
    if np.sum(valid_mask) < 10:
        return _empty_result("insufficient_valid_points_above_substrate")

    X_fit = X_all[valid_mask]
    Z_fit = Z_all[valid_mask]

    # 4. Initial parameter estimates
    # Apex curvature R0 estimation: circle fit on apex cap (Z <= 0.4 * H)
    cap_mask = valid_mask & (Z_all <= 0.4 * height_apex)
    if np.sum(cap_mask) >= 5:
        try:
            _, r0_fit_val = fit_circle(xy[cap_mask])
            r0_init = float(r0_fit_val)
        except Exception:
            r0_init = float(height_apex)
    else:
        r0_init = float(height_apex)

    if not np.isfinite(r0_init) or r0_init <= 5.0:
        # Geometric spherical cap fallback: R = (W^2 + 4H^2) / (8H)
        width_approx = float(np.max(X_fit) - np.min(X_fit))
        r0_init = float((width_approx**2 + 4.0 * height_apex**2) / (8.0 * height_apex))
        r0_init = max(10.0, r0_init)

    # Bond number initial guess from physics if provided
    bo_init = 0.05
    delta_rho = 998.2
    g_acc = 9.80665
    if physics:
        rho1 = float(physics.get("rho1", 1000.0))
        rho2 = float(physics.get("rho2", 1.2))
        delta_rho = max(1.0, abs(rho1 - rho2))
        g_acc = float(physics.get("g", 9.80665))
        gamma_guess = float(physics.get("gamma", 0.0728))
        if gamma_guess > 0 and px_per_mm > 0:
            r0_m = (r0_init / px_per_mm) * 1e-3
            bo_init = float(np.clip((delta_rho * g_acc * r0_m**2) / gamma_guess, 0.001, 0.3))

    # 5. Non-linear least-squares optimization
    # Parameters: [R0, Bo, dX, dZ] if optimize_bo else [R0, dX, dZ]
    if optimize_bo:
        p0 = [r0_init, bo_init, 0.0, 0.0]
        lb = [max(5.0, r0_init * 0.2), -0.1, -30.0, -30.0]
        ub = [r0_init * 5.0, 0.4, 30.0, 30.0]

        def residuals(p: list[float]) -> np.ndarray:
            r0_val, bo_val, dx_val, dz_val = p
            xs = X_fit - dx_val
            zs = Z_fit - dz_val
            # Polar angle from apex center of curvature (0, R0)
            # z_from_center = r0 - z is positive for z < r0 (acute) and negative for z > r0 (obtuse)
            z_from_center = r0_val - zs
            alphas = np.arctan2(np.abs(xs), z_from_center)
            dists = np.sqrt(xs**2 + z_from_center**2)
            r_model = radius_lbadsa(alphas, r0_val, bo_val)
            return dists - r_model

    else:
        p0 = [r0_init, 0.0, 0.0]
        lb = [max(5.0, r0_init * 0.2), -30.0, -30.0]
        ub = [r0_init * 5.0, 30.0, 30.0]

        def residuals(p: list[float]) -> np.ndarray:
            r0_val, dx_val, dz_val = p
            xs = X_fit - dx_val
            zs = Z_fit - dz_val
            z_from_center = r0_val - zs
            alphas = np.arctan2(np.abs(xs), z_from_center)
            dists = np.sqrt(xs**2 + z_from_center**2)
            r_model = radius_lbadsa(alphas, r0_val, bo_init)
            return dists - r_model

    try:
        res = least_squares(
            residuals,
            p0,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=60,
        )
        if optimize_bo:
            r0_fit, bo_fit, dx_fit, dz_fit = res.x
        else:
            r0_fit, dx_fit, dz_fit = res.x
            bo_fit = bo_init

        res_vals = res.fun
        rmse_px = float(np.sqrt(np.mean(res_vals**2)))
        max_residual_px = float(np.max(np.abs(res_vals)))
        fit_success = bool(res.success)
    except Exception:
        r0_fit = r0_init
        bo_fit = bo_init
        dx_fit = dz_fit = 0.0
        rmse_px = float("nan")
        max_residual_px = float("nan")
        fit_success = False

    # Refined apex in image coordinates
    apex_refined = apex_pt + dx_fit * x_hat + dz_fit * z_hat

    # 6. Contact point heights and angles
    # Determine contact point locations
    if contact_points is not None:
        p_left = np.asarray(contact_points[0], dtype=float)
        p_right = np.asarray(contact_points[1], dtype=float)
    else:
        # Intersect baseline with drop width
        half_w = 0.5 * float(np.max(X_fit) - np.min(X_fit))
        p_left = apex_refined - half_w * x_hat + height_apex * z_hat
        p_right = apex_refined + half_w * x_hat + height_apex * z_hat

    h_left = float(abs(np.dot(p_left - apex_refined, z_hat)))
    h_right = float(abs(np.dot(p_right - apex_refined, z_hat)))

    theta_left_deg, alpha_c_left = contact_angle_lbadsa(r0_fit, bo_fit, h_left)
    theta_right_deg, alpha_c_right = contact_angle_lbadsa(r0_fit, bo_fit, h_right)

    # Correct for local substrate tilt/tangent if curved substrate
    if sub_profile is not None:
        slope_left = sub_profile.eval_tangent_angle_deg(float(p_left[0]), float(p_left[1]))
        slope_right = sub_profile.eval_tangent_angle_deg(float(p_right[0]), float(p_right[1]))
    else:
        slope_left = slope_right = 0.0

    # Intrinsic contact angle is with respect to the local substrate tangent
    theta_left_deg = float(np.clip(theta_left_deg, 1.0, 179.0))
    theta_right_deg = float(np.clip(theta_right_deg, 1.0, 179.0))
    theta_mean_deg = float(0.5 * (theta_left_deg + theta_right_deg))

    # 7. Physical quantities
    r0_mm = float(r0_fit / px_per_mm) if px_per_mm > 0 else float("nan")
    gamma_mN_m = surface_tension_from_bo(bo_fit, r0_mm, delta_rho=delta_rho, g=g_acc)

    # 8. Reconstruct fitted model curve in image coordinates for overlay
    max_alpha = max(alpha_c_left, alpha_c_right, np.radians(theta_mean_deg))
    max_alpha = min(np.radians(165.0), max(np.radians(15.0), max_alpha))
    alphas_curve = np.linspace(-max_alpha, max_alpha, 120)

    x_mod, z_mod = cartesian_lbadsa(np.abs(alphas_curve), r0_fit, bo_fit)
    x_mod_signed = np.where(alphas_curve >= 0, x_mod, -x_mod)

    # Transform back to image coordinates
    model_pts = apex_refined + np.outer(x_mod_signed, x_hat) + np.outer(z_mod, z_hat)

    # 9. Diagnostics
    diagnostics = {
        "fit_success": fit_success,
        "rmse_px": rmse_px,
        "max_residual_px": max_residual_px,
        "R0_px": float(r0_fit),
        "R0_mm": float(r0_mm),
        "bo": float(bo_fit),
        "gamma_mN_m": float(gamma_mN_m),
        "dx_apex_px": float(dx_fit),
        "dz_apex_px": float(dz_fit),
        "alpha_c_left_deg": float(np.degrees(alpha_c_left)),
        "alpha_c_right_deg": float(np.degrees(alpha_c_right)),
        "valid_bond_range": bool(0.0005 <= bo_fit <= 0.35),
        "slope_substrate_left_deg": float(slope_left),
        "slope_substrate_right_deg": float(slope_right),
    }

    return {
        "theta_left_deg": theta_left_deg,
        "theta_right_deg": theta_right_deg,
        "theta_mean_deg": theta_mean_deg,
        "contact_angle_deg": theta_mean_deg,
        "bond_number": float(bo_fit),
        "R0_px": float(r0_fit),
        "R0_mm": float(r0_mm),
        "surface_tension_mN_m": float(gamma_mN_m),
        "rmse_px": rmse_px,
        "max_residual_px": max_residual_px,
        "model_contour_xy": model_pts,
        "diagnostics": diagnostics,
    }


def _empty_result(reason: str) -> dict[str, Any]:
    """Return fallback payload when fitting cannot execute."""
    return {
        "theta_left_deg": float("nan"),
        "theta_right_deg": float("nan"),
        "theta_mean_deg": float("nan"),
        "contact_angle_deg": float("nan"),
        "bond_number": float("nan"),
        "R0_px": float("nan"),
        "R0_mm": float("nan"),
        "surface_tension_mN_m": float("nan"),
        "rmse_px": float("nan"),
        "max_residual_px": float("nan"),
        "model_contour_xy": np.empty((0, 2), dtype=float),
        "diagnostics": {"fit_success": False, "reason": reason},
    }
