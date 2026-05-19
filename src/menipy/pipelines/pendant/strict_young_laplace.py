"""Strict, unit-consistent Young-Laplace fitting for pendant drops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from menipy.models.surface_tension import surface_tension


@dataclass(frozen=True)
class PendantStrictFitInput:
    """Inputs for the strict pendant Young-Laplace fit."""

    contour_px: np.ndarray
    axis_x_px: float
    apex_y_px: float
    px_per_mm: float
    r0_seed_mm: float
    beta_seed: float
    physics: dict[str, Any]
    needle_radius_mm: float | None = None


def build_pendant_profile_envelope_mm(
    contour_px: np.ndarray,
    *,
    axis_x_px: float,
    apex_y_px: float,
    px_per_mm: float,
    bin_px: float = 1.0,
) -> np.ndarray:
    """Collapse a pendant contour into a radial ``(r_mm, z_mm)`` envelope."""
    xy = np.asarray(contour_px, dtype=float).reshape(-1, 2)
    if xy.shape[0] < 3 or px_per_mm <= 0:
        return np.empty((0, 2), dtype=float)

    bin_px = max(float(bin_px), 1.0)
    row_keys = np.rint(xy[:, 1] / bin_px).astype(int)
    rows: list[tuple[float, float]] = []
    for row in np.unique(row_keys):
        pts = xy[row_keys == row]
        if pts.size == 0:
            continue
        y_px = float(np.mean(pts[:, 1]))
        z_mm = (float(apex_y_px) - y_px) / float(px_per_mm)
        if z_mm < -0.5 / float(px_per_mm):
            continue
        xs = pts[:, 0]
        if xs.size >= 2:
            r_mm = (float(np.max(xs)) - float(np.min(xs))) / (2.0 * px_per_mm)
        else:
            r_mm = float(np.max(np.abs(xs - float(axis_x_px)))) / float(px_per_mm)
        if np.isfinite(z_mm) and np.isfinite(r_mm) and r_mm >= 0:
            rows.append((r_mm, max(0.0, z_mm)))

    if not rows:
        return np.empty((0, 2), dtype=float)

    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 1])]
    merged: list[tuple[float, float]] = []
    for z in np.unique(arr[:, 1]):
        r = float(np.max(arr[arr[:, 1] == z, 0]))
        merged.append((r, float(z)))
    profile = np.asarray(merged, dtype=float)
    if profile.shape[0] < 2:
        return np.empty((0, 2), dtype=float)

    if profile[0, 1] > 1e-9:
        profile = np.vstack([[0.0, 0.0], profile])
    else:
        profile[0, 1] = 0.0
        profile[0, 0] = min(profile[0, 0], profile[1, 0] if profile.shape[0] > 1 else 0.0)
    return profile


def pendant_contour_to_model_mm(
    contour_px: np.ndarray,
    *,
    axis_x_px: float,
    apex_y_px: float,
    px_per_mm: float,
) -> np.ndarray:
    """Convert image-space pendant contour points into apex-centered mm."""
    xy = np.asarray(contour_px, dtype=float).reshape(-1, 2)
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    x_mm = (xy[:, 0] - float(axis_x_px)) / float(px_per_mm)
    z_mm = (float(apex_y_px) - xy[:, 1]) / float(px_per_mm)
    return np.column_stack([x_mm, z_mm])


def model_mm_to_pendant_px(
    model_mm: np.ndarray,
    *,
    axis_x_px: float,
    apex_y_px: float,
    px_per_mm: float,
) -> np.ndarray:
    """Convert apex-centered strict model coordinates back to image pixels."""
    xy = np.asarray(model_mm, dtype=float).reshape(-1, 2)
    x_px = xy[:, 0] * float(px_per_mm) + float(axis_x_px)
    y_px = float(apex_y_px) - xy[:, 1] * float(px_per_mm)
    return np.column_stack([x_px, y_px])


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
        r, z, psi = y
        if abs(r) < 1e-10:
            sin_psi_over_r = 1.0
        else:
            sin_psi_over_r = float(np.sin(psi) / r)
        return [
            float(np.cos(psi)),
            float(np.sin(psi)),
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


def _normal_projection_residuals_mm(obs_mm: np.ndarray, model_mm: np.ndarray) -> np.ndarray:
    """Project observed points to nearest model normals and return mm distances."""
    if model_mm.shape[0] < 3:
        return np.full(obs_mm.shape[0], 1e3, dtype=float)

    tangents = np.gradient(model_mm, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    _, idx = cKDTree(model_mm).query(obs_mm)
    return np.einsum("ij,ij->i", obs_mm - model_mm[idx], normals[idx]).astype(float)


def _bounds_from_seed(
    *,
    r0_seed_mm: float,
    beta_seed: float,
    diameter_mm: float,
    height_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r0_seed_mm = max(float(r0_seed_mm), 0.05)
    beta_seed = float(np.clip(beta_seed, 0.02, 4.0))
    r0_upper = max(20.0, r0_seed_mm * 5.0, diameter_mm * 3.0, height_mm * 3.0)
    offset_x = max(0.25, diameter_mm * 0.15)
    offset_z = max(0.25, height_mm * 0.15)
    lower = np.array([max(0.02, r0_seed_mm / 5.0), 1e-3, -offset_x, -offset_z])
    upper = np.array([r0_upper, 5.0, offset_x, offset_z])
    x0 = np.array([r0_seed_mm, beta_seed, 0.0, 0.0])
    return x0, lower, upper


def _is_pinned(params: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    span = np.maximum(upper - lower, 1e-12)
    tol = np.maximum(1e-5, span * 1e-3)
    return bool(np.any(params <= lower + tol) or np.any(params >= upper - tol))


def fit_pendant_young_laplace_strict(fit_input: PendantStrictFitInput) -> dict[str, Any]:
    """Fit a calibrated pendant contour to a strict Young-Laplace profile."""
    obs_mm = pendant_contour_to_model_mm(
        fit_input.contour_px,
        axis_x_px=fit_input.axis_x_px,
        apex_y_px=fit_input.apex_y_px,
        px_per_mm=fit_input.px_per_mm,
    )
    if obs_mm.shape[0] < 8:
        return {
            "params": [],
            "param_names": ["r0_mm", "beta", "x_offset_mm", "z_offset_mm"],
            "residuals": {"rmse": float("nan"), "max_abs": float("nan"), "dof": 0, "r": []},
            "solver": {
                "backend": "scipy.least_squares",
                "method": "trf",
                "iterations": 0,
                "success": False,
                "message": "not enough contour points",
            },
            "strict_fit_success": False,
            "strict_fit_warning": "not_enough_contour_points",
        }

    obs_mm = obs_mm[obs_mm[:, 1] >= -0.5 / float(fit_input.px_per_mm)]
    envelope_mm = build_pendant_profile_envelope_mm(
        fit_input.contour_px,
        axis_x_px=fit_input.axis_x_px,
        apex_y_px=fit_input.apex_y_px,
        px_per_mm=fit_input.px_per_mm,
    )
    diameter_mm = float(np.ptp(obs_mm[:, 0]))
    height_mm = float(np.ptp(obs_mm[:, 1]))
    if envelope_mm.shape[0] >= 3:
        diameter_mm = max(diameter_mm, float(2.0 * np.max(envelope_mm[:, 0])))
        height_mm = max(height_mm, float(np.max(envelope_mm[:, 1])))
    x0, lower, upper = _bounds_from_seed(
        r0_seed_mm=fit_input.r0_seed_mm,
        beta_seed=fit_input.beta_seed,
        diameter_mm=diameter_mm,
        height_mm=height_mm,
    )

    def model_from_params(params: np.ndarray) -> np.ndarray:
        r0_mm, beta, x_offset_mm, z_offset_mm = params
        model = integrate_young_laplace_profile_mm(
            r0_mm,
            beta,
            target_height_mm=height_mm,
        )
        if model.size == 0:
            return model
        return model + np.array([x_offset_mm, z_offset_mm])

    def residuals(params: np.ndarray) -> np.ndarray:
        model = model_from_params(params)
        r = _normal_projection_residuals_mm(obs_mm, model)
        # Keep small offset freedom, but discourage using offsets as a full shape fit.
        r0_mm, _beta, x_offset_mm, z_offset_mm = params
        offset_scale = max(0.05, r0_mm * 0.25)
        regularization = np.array(
            [x_offset_mm / offset_scale, z_offset_mm / offset_scale], dtype=float
        ) * 0.01
        return np.concatenate([r, regularization])

    res = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.03,
        max_nfev=300,
        method="trf",
    )

    r_vec = residuals(res.x)
    contour_r = r_vec[: obs_mm.shape[0]]
    rmse = float(np.sqrt(np.mean(contour_r**2))) if contour_r.size else float("nan")
    max_abs = float(np.max(np.abs(contour_r))) if contour_r.size else float("nan")
    model_mm = model_from_params(res.x)
    coverage_height_mm = float(np.ptp(model_mm[:, 1])) if model_mm.size else 0.0
    radial_profile_mm, radial_meta = integrate_young_laplace_profile_mm(
        float(res.x[0]),
        float(res.x[1]),
        target_height_mm=height_mm,
        branch="right",
        return_metadata=True,
    )

    rho1 = float(fit_input.physics.get("rho1", 1000.0))
    rho2 = float(fit_input.physics.get("rho2", 1.2))
    g = float(fit_input.physics.get("g", 9.80665))
    r0_mm = float(res.x[0])
    beta = float(res.x[1])
    gamma_mn_m = float(surface_tension(rho1 - rho2, g, r0_mm, beta) * 1000.0)

    threshold_mm = max(0.05, 0.03 * max(diameter_mm, 1e-12))
    params_finite = bool(np.all(np.isfinite(res.x)))
    positive_params = bool(r0_mm > 0 and beta > 0)
    coverage_ok = bool(coverage_height_mm >= 0.98 * height_mm)
    pinned = _is_pinned(res.x, lower, upper)
    strict_success = bool(
        res.success
        and params_finite
        and positive_params
        and coverage_ok
        and not pinned
        and np.isfinite(rmse)
        and rmse <= threshold_mm
    )

    warning = None
    if not res.success:
        warning = "optimizer_failed"
    elif not params_finite or not positive_params:
        warning = "invalid_fit_parameters"
    elif pinned:
        warning = "fit_parameter_at_bound"
    elif not coverage_ok:
        warning = "model_height_coverage_failed"
    elif not np.isfinite(rmse) or rmse > threshold_mm:
        warning = "residual_gate_failed"

    model_px = model_mm_to_pendant_px(
        model_mm,
        axis_x_px=fit_input.axis_x_px,
        apex_y_px=fit_input.apex_y_px,
        px_per_mm=fit_input.px_per_mm,
    )

    return {
        "params": res.x.tolist(),
        "param_names": ["r0_mm", "beta", "x_offset_mm", "z_offset_mm"],
        "residuals": {
            "rmse": rmse,
            "max_abs": max_abs,
            "dof": int(contour_r.size - res.x.size),
            "r": contour_r.tolist(),
            "units": "mm",
            "threshold_mm": threshold_mm,
        },
        "solver": {
            "backend": "scipy.least_squares",
            "method": "trf",
            "iterations": int(res.nfev),
            "success": bool(res.success),
            "message": str(res.message),
        },
        "strict_fit_success": strict_success,
        "strict_fit_warning": warning,
        "strict_surface_tension_mN_m": gamma_mn_m,
        "strict_r0_mm": r0_mm,
        "strict_beta": beta,
        "strict_x_offset_mm": float(res.x[2]),
        "strict_z_offset_mm": float(res.x[3]),
        "strict_rmse_mm": rmse,
        "strict_residual_threshold_mm": threshold_mm,
        "strict_fit_stop_reason": radial_meta.get("stop_reason", "unknown"),
        "strict_model_coverage_height_mm": coverage_height_mm,
        "strict_observed_height_mm": height_mm,
        "strict_observed_diameter_mm": diameter_mm,
        "observed_profile_mm": envelope_mm.tolist(),
        "model_radial_profile_mm": radial_profile_mm.tolist(),
        "model_profile_mm": model_mm.tolist(),
        "model_profile_px": model_px.tolist(),
    }
