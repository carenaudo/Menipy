"""Hydrodynamic contact line extrapolation and tilting plate droplet retention models.

Academic References:
    1. Cox, R. G. (1986).
       "The dynamics of the spreading of liquids on a solid surface. Part 1. Viscous flow."
       Journal of Fluid Mechanics, 131, 1-46.
       DOI: 10.1017/S0022112086000032

    2. Voinov, O. V. (1976).
       "Hydrodynamics of wetting."
       Fluid Dynamics, 11(5), 714-721.
       DOI: 10.1007/BF01012963

    3. Furmidge, C. G. L. (1962).
       "Studies at interfaces. I. The sliding of liquid drops on solid surfaces and a theory for spray retention."
       Journal of Colloid Science, 17(4), 309-324.
       DOI: 10.1016/0095-8522(62)90011-9

    4. Bonn, D., Eggers, J., Indekeu, J., Meunier, J., & Rolley, E. (2009).
       "Wetting and spreading."
       Reviews of Modern Physics, 81(2), 739-800.
       DOI: 10.1103/RevModPhys.81.739

    5. Snoeijer, J. H., & Andreotti, B. (2013).
       "Moving contact lines: scales, regimes, and dynamical transitions."
       Annual Review of Fluid Mechanics, 45, 269-292.
       DOI: 10.1146/annurev-fluid-011212-140734

Attribution & Clean-Room Implementation:
    Independent clean-room Python/NumPy implementation authored for Menipy under MIT license.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def cox_voinov_extrapolation(
    velocities_mm_s: Sequence[float],
    angles_deg: Sequence[float],
    *,
    viscosity_pa_s: float | None = None,
    surface_tension_mN_m: float | None = None,
) -> dict[str, Any]:
    """Fit Cox-Voinov hydrodynamic contact angle equation and extrapolate to zero velocity.

    Equation:
        \\theta_d^3 = \\theta_0^3 \\pm 9 Ca \\ln(L / \\ell_m)
        \\theta_d^3 = \\theta_0^3 + k \\cdot v_{CL}

    Args:
        velocities_mm_s: Contact line velocities in mm/s.
        angles_deg: Dynamic contact angles in degrees.
        viscosity_pa_s: Optional dynamic viscosity mu in Pa*s (e.g. 0.001 for water).
        surface_tension_mN_m: Optional surface tension gamma in mN/m (e.g. 72.8 for water).

    Returns:
        dict containing:
            - theta_0_deg: Extrapolated equilibrium/static contact angle in degrees.
            - slope_rad3_per_mm_s: Fitted slope k in rad^3 / (mm/s).
            - r_squared: Linear regression coefficient of determination.
            - n_points: Number of valid points fitted.
            - ln_L_over_lm: Microscopic length ratio parameter (if mu, gamma provided).
            - mean_capillary_number: Average Ca across points (if mu, gamma provided).
    """
    v = np.asarray(velocities_mm_s, dtype=float)
    theta = np.asarray(angles_deg, dtype=float)

    valid = np.isfinite(v) & np.isfinite(theta) & (theta > 0.0) & (theta < 180.0)
    v = v[valid]
    theta = theta[valid]

    if len(v) < 3 or float(np.ptp(v)) <= 1e-6:
        # Insufficient velocity variation for extrapolation
        theta_median = float(np.median(theta)) if len(theta) > 0 else float("nan")
        return {
            "theta_0_deg": theta_median,
            "slope_rad3_per_mm_s": 0.0,
            "r_squared": 0.0,
            "n_points": len(v),
            "ln_L_over_lm": None,
            "mean_capillary_number": None,
        }

    # Dependent variable y = theta^3 in radians^3
    theta_rad = np.radians(theta)
    y = theta_rad**3

    # Linear least squares: y = a + b * v
    design = np.column_stack([np.ones_like(v), v])
    coef, residuals, rank, s = np.linalg.lstsq(design, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    # R^2 calculation
    y_pred = a + b * v
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r_squared = float(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 1e-12 else 1.0

    # theta_0 = a^(1/3)
    if a > 0:
        theta_0_rad = a ** (1.0 / 3.0)
        theta_0_deg = float(np.degrees(theta_0_rad))
    else:
        # Fallback to minimum observed angle if intercept is non-positive
        theta_0_deg = float(np.min(theta))

    # Physical parameters if viscosity and surface tension are provided
    ln_L_over_lm = None
    mean_ca = None
    if (
        viscosity_pa_s is not None
        and viscosity_pa_s > 0
        and surface_tension_mN_m is not None
        and surface_tension_mN_m > 0
    ):
        gamma_N_m = surface_tension_mN_m * 1e-3
        # v in mm/s -> v in m/s: multiply by 1e-3
        # b in rad^3 / (mm/s) -> b_SI in rad^3 / (m/s) = b * 1e3
        b_SI = b * 1e3
        # b_SI = 9 * (mu / gamma) * ln(L / lm)
        # ln(L / lm) = b_SI * gamma / (9 * mu)
        ln_val = (abs(b_SI) * gamma_N_m) / (9.0 * viscosity_pa_s)
        ln_L_over_lm = float(ln_val)
        v_m_s = np.abs(v) * 1e-3
        ca_values = (viscosity_pa_s * v_m_s) / gamma_N_m
        mean_ca = float(np.mean(ca_values))

    return {
        "theta_0_deg": theta_0_deg,
        "slope_rad3_per_mm_s": b,
        "r_squared": r_squared,
        "n_points": len(v),
        "ln_L_over_lm": ln_L_over_lm,
        "mean_capillary_number": mean_ca,
    }


def furmidge_retention_force(
    theta_adv_deg: float,
    theta_rec_deg: float,
    contact_width_mm: float,
    surface_tension_mN_m: float,
    *,
    droplet_mass_mg: float | None = None,
    g: float = 9.80665,
) -> dict[str, float | None]:
    """Calculate droplet retention force on an inclined substrate using Furmidge relation.

    Equation:
        f_{retention} = gamma (cos theta_R - cos theta_A)  [mN/m]
        F_{retention} = gamma w (cos theta_R - cos theta_A)  [uN]

    Args:
        theta_adv_deg: Advancing contact angle in degrees.
        theta_rec_deg: Receding contact angle in degrees.
        contact_width_mm: Droplet contact base width w in mm.
        surface_tension_mN_m: Surface tension gamma in mN/m.
        droplet_mass_mg: Droplet mass m in milligrams (optional).
        g: Gravitational acceleration in m/s^2.

    Returns:
        dict containing:
            - retention_force_per_width_mN_m: Line retention force f in mN/m.
            - total_retention_force_uN: Total lateral retention force F in micro-Newtons.
            - cos_diff: Dimensionless cosine difference (cos(theta_R) - cos(theta_A)).
            - critical_sliding_angle_deg: Predicted critical tilt angle alpha_{crit} in degrees.
    """
    theta_A = math.radians(float(theta_adv_deg))
    theta_R = math.radians(float(theta_rec_deg))
    w_mm = max(1e-6, float(contact_width_mm))
    gamma = max(1e-6, float(surface_tension_mN_m))

    cos_diff = float(math.cos(theta_R) - math.cos(theta_A))
    f_ret = float(gamma * cos_diff)
    # F = gamma [mN/m] * w [mm] * cos_diff = gamma * 1e-3 * w * 1e-3 * cos_diff [N]
    #   = gamma * w * cos_diff [1e-6 N] = gamma * w * cos_diff [uN]
    f_total_uN = float(gamma * w_mm * cos_diff)

    alpha_crit_deg: float | None = None
    if droplet_mass_mg is not None and droplet_mass_mg > 0:
        # Gravity force along slope: F_g = m * g * sin(alpha)
        # m in mg = m * 1e-6 kg
        # m * g in N = m * g * 1e-6 N = m * g [uN]
        f_g_uN = droplet_mass_mg * g
        sin_alpha = f_total_uN / max(1e-9, f_g_uN)
        if 0.0 <= sin_alpha <= 1.0:
            alpha_crit_deg = float(math.degrees(math.asin(sin_alpha)))
        elif sin_alpha > 1.0:
            alpha_crit_deg = 90.0  # Pinned (retention exceeds drop weight)
        else:
            alpha_crit_deg = 0.0

    return {
        "retention_force_per_width_mN_m": f_ret,
        "total_retention_force_uN": f_total_uN,
        "cos_diff": cos_diff,
        "critical_sliding_angle_deg": alpha_crit_deg,
    }


def analyze_tilting_plate(
    frames: Sequence[Any],
    *,
    surface_tension_mN_m: float | None = None,
    droplet_mass_mg: float | None = None,
) -> dict[str, Any]:
    """Analyze a tilting plate sequence to detect critical sliding angle and retention force.

    Args:
        frames: Sequence of TemporalFrameResult objects.
        surface_tension_mN_m: Optional surface tension in mN/m.
        droplet_mass_mg: Optional droplet mass in mg.

    Returns:
        dict containing:
            - is_tilting: True if substrate baseline angle changed by > 1.0 deg.
            - tilt_range_deg: Total inclination angle range in degrees.
            - critical_sliding_angle_deg: Measured plate angle at sliding onset.
            - critical_frame_index: Frame index where sliding began.
            - theta_advancing_critical_deg: Downhill angle at sliding onset.
            - theta_receding_critical_deg: Uphill angle at sliding onset.
            - retention_metrics: Furmidge retention force metrics at sliding onset.
    """
    valid_frames = [f for f in frames if getattr(f, "accepted", False) and getattr(f, "baseline", None)]
    if len(valid_frames) < 3:
        return {"is_tilting": False, "tilt_range_deg": 0.0}

    # Compute baseline angles
    angles = []
    for f in valid_frames:
        (x1, y1), (x2, y2) = f.baseline
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    tilt_range = float(np.ptp(angles))
    is_tilting = bool(tilt_range >= 1.0)
    if not is_tilting:
        return {"is_tilting": False, "tilt_range_deg": tilt_range}

    initial_angle = angles[0]
    relative_tilts = [abs(a - initial_angle) for a in angles]

    # Find sliding onset: first frame transitioning from pinned to moving
    critical_frame = None
    critical_tilt = None
    for f, tilt in zip(valid_frames, relative_tilts):
        state = getattr(f, "state", "pinned")
        vel = getattr(f, "contact_velocity_mm_s", None)
        if state in ("advancing", "receding") or (vel is not None and abs(vel) > 0.05):
            critical_frame = f
            critical_tilt = tilt
            break

    if critical_frame is None:
        # Pinned throughout entire tilt
        critical_tilt = max(relative_tilts)
        critical_frame = valid_frames[-1]

    left = getattr(critical_frame, "theta_left_deg", None)
    right = getattr(critical_frame, "theta_right_deg", None)
    half_w = getattr(critical_frame, "half_width_mm", None)

    # In tilt, the downhill contact angle is max(left, right), uphill is min(left, right)
    theta_adv = max(left, right) if left is not None and right is not None else None
    theta_rec = min(left, right) if left is not None and right is not None else None
    width_mm = (2.0 * half_w) if half_w is not None else 2.0

    retention_metrics = None
    if (
        surface_tension_mN_m is not None
        and surface_tension_mN_m > 0
        and theta_adv is not None
        and theta_rec is not None
    ):
        retention_metrics = furmidge_retention_force(
            theta_adv_deg=theta_adv,
            theta_rec_deg=theta_rec,
            contact_width_mm=width_mm,
            surface_tension_mN_m=surface_tension_mN_m,
            droplet_mass_mg=droplet_mass_mg,
        )

    return {
        "is_tilting": True,
        "tilt_range_deg": tilt_range,
        "critical_sliding_angle_deg": float(critical_tilt) if critical_tilt is not None else None,
        "critical_frame_index": int(critical_frame.frame_index) if critical_frame else None,
        "theta_advancing_critical_deg": float(theta_adv) if theta_adv is not None else None,
        "theta_receding_critical_deg": float(theta_rec) if theta_rec is not None else None,
        "retention_metrics": retention_metrics,
    }
