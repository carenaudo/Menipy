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
    max_step: float = 0.02,
) -> np.ndarray:
    """Integrate a symmetric pendant Young-Laplace profile in millimetres."""
    r0_mm = float(r0_mm)
    beta = float(beta)
    if not np.isfinite(r0_mm) or not np.isfinite(beta) or r0_mm <= 0:
        return np.empty((0, 2), dtype=float)

    z_target = None
    if target_height_mm is not None and target_height_mm > 0:
        z_target = max(float(target_height_mm) / r0_mm * 1.1, 0.2)

    def ode(_s: float, y: np.ndarray) -> list[float]:
        r, z, psi = y
        if abs(r) < 1e-10:
            sin_psi_over_r = 1.0
        else:
            sin_psi_over_r = float(np.sin(psi) / r)
        return [
            float(np.cos(psi)),
            float(np.sin(psi)),
            float(2.0 + beta * z - sin_psi_over_r),
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
        return np.empty((0, 2), dtype=float)

    r_right = sol.y[0] * r0_mm
    z_right = sol.y[1] * r0_mm
    r_left = -r_right[::-1]
    z_left = z_right[::-1]
    r_full = np.concatenate([r_left[:-1], r_right])
    z_full = np.concatenate([z_left[:-1], z_right])
    return np.column_stack([r_full, z_full])


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

    diameter_mm = float(np.ptp(obs_mm[:, 0]))
    height_mm = float(np.ptp(obs_mm[:, 1]))
    x0, lower, upper = _bounds_from_seed(
        r0_seed_mm=fit_input.r0_seed_mm,
        beta_seed=fit_input.beta_seed,
        diameter_mm=diameter_mm,
        height_mm=height_mm,
    )

    def model_from_params(params: np.ndarray) -> np.ndarray:
        r0_mm, beta, x_offset_mm, z_offset_mm = params
        model = integrate_young_laplace_profile_mm(
            r0_mm, beta, target_height_mm=height_mm + abs(float(z_offset_mm))
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
        "strict_model_coverage_height_mm": coverage_height_mm,
        "strict_observed_height_mm": height_mm,
        "strict_observed_diameter_mm": diameter_mm,
        "model_profile_mm": model_mm.tolist(),
        "model_profile_px": model_px.tolist(),
    }
