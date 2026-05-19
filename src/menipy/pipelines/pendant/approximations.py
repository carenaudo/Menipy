"""Built-in pendant surface-tension approximation plugins."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from menipy.common.registry import register_pendant_approximator
from menipy.models.surface_tension import surface_tension
from menipy.pipelines.pendant.strict_young_laplace import (
    integrate_young_laplace_profile_mm,
)


DEFAULT_SELECTED_PLANES = (0.6, 0.7, 0.8, 0.9, 1.0)


def _profile(profile_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile = np.asarray(profile_mm, dtype=float).reshape(-1, 2)
    if profile.shape[0] < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    profile = profile[np.all(np.isfinite(profile), axis=1)]
    profile = profile[(profile[:, 0] >= 0) & (profile[:, 1] >= 0)]
    if profile.shape[0] < 3:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(profile[:, 1])
    z = profile[order, 1]
    r = profile[order, 0]
    z, idx = np.unique(z, return_index=True)
    return r[idx], z


def _physics(physics: dict[str, Any]) -> tuple[float, float, float]:
    rho1 = float(physics.get("rho1", 1000.0))
    rho2 = float(physics.get("rho2", 1.2))
    g = float(physics.get("g", 9.80665))
    return rho1 - rho2, g, rho1


def _r0_mm_from_context(ctx: Any) -> float | None:
    results = getattr(ctx, "results", {}) or {}
    for key in ("geometric_r0_mm", "r0_mm", "strict_r0_mm"):
        value = results.get(key)
        if value is not None and np.isfinite(float(value)) and float(value) > 0:
            return float(value)
    fit = getattr(ctx, "fit", {}) or {}
    value = fit.get("strict_r0_mm")
    if value is not None and np.isfinite(float(value)) and float(value) > 0:
        return float(value)
    return None


def _surface_tension_from_h(delta_rho: float, g: float, de_mm: float, h: float) -> float:
    de_m = de_mm / 1000.0
    return delta_rho * g * de_m**2 / h


def _beta_from_gamma(delta_rho: float, g: float, r0_mm: float | None, gamma_n_m: float) -> float | None:
    if r0_mm is None or gamma_n_m <= 0:
        return None
    return float(delta_rho * g * (r0_mm / 1000.0) ** 2 / gamma_n_m)


@lru_cache(maxsize=1)
def _selected_plane_lookup_all() -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    beta_grid = np.linspace(0.03, 2.5, 24)
    height_grid = np.linspace(0.8, 4.5, 24)
    by_plane = {
        round(float(k), 4): ([], [], []) for k in DEFAULT_SELECTED_PLANES + (1.0,)
    }
    for beta in beta_grid:
        for height in height_grid:
            model = integrate_young_laplace_profile_mm(
                1.0,
                float(beta),
                target_height_mm=float(height),
                branch="right",
                max_step=0.08,
            )
            r, z = _profile(model)
            if r.size < 3:
                continue
            de = float(2.0 * np.max(r))
            if de <= 0:
                continue
            for k, values in by_plane.items():
                z_plane = float(k) * de
                if z_plane > float(np.max(z)):
                    continue
                ds = 2.0 * float(np.interp(z_plane, z, r))
                values[0].append(ds / de)
                values[1].append(float(np.max(z)) / de)
                values[2].append(float(beta) * de**2)

    return {
        k: (
            np.asarray(values[0], dtype=float),
            np.asarray(values[1], dtype=float),
            np.asarray(values[2], dtype=float),
        )
        for k, values in by_plane.items()
    }


def _selected_plane_lookup(k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = (np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float))
    return _selected_plane_lookup_all().get(round(float(k), 4), empty)


def _lookup_h_from_s(k: float, s: float, height_over_de: float) -> tuple[float | None, str]:
    s_arr, hde_arr, h_arr = _selected_plane_lookup(round(float(k), 4))
    if s_arr.size < 3 or not np.isfinite(s):
        return None, "lookup_unavailable"
    score = ((s_arr - float(s)) / 0.04) ** 2 + (
        (hde_arr - float(height_over_de)) / 0.08
    ) ** 2
    idx = int(np.argmin(score))
    if not np.isfinite(score[idx]) or score[idx] > 25.0:
        return None, "outside_lookup_range"
    return float(h_arr[idx]), "ok"


def _selected_plane_estimate(
    ctx: Any, profile_mm: np.ndarray, physics: dict[str, Any], *, k: float
) -> dict[str, Any]:
    r, z = _profile(profile_mm)
    prefix = "approx_selected_plane"
    if r.size < 3:
        return {f"{prefix}_status": "not_enough_profile_points"}

    de_mm = float(2.0 * np.max(r))
    z_plane = float(k) * de_mm
    if z_plane > float(np.max(z)):
        return {
            f"{prefix}_status": "unavailable_plane_outside_drop",
            f"{prefix}_k": float(k),
            f"{prefix}_diameter_mm": de_mm,
        }

    ds_mm = float(2.0 * np.interp(z_plane, z, r))
    s_ratio = ds_mm / de_mm if de_mm > 0 else float("nan")
    height_over_de = float(np.max(z)) / de_mm if de_mm > 0 else float("nan")
    h, status = _lookup_h_from_s(k, s_ratio, height_over_de)
    out = {
        f"{prefix}_status": status,
        f"{prefix}_k": float(k),
        f"{prefix}_diameter_mm": de_mm,
        f"{prefix}_section_diameter_mm": ds_mm,
        f"{prefix}_s": s_ratio,
        f"{prefix}_height_over_diameter": height_over_de,
    }
    if h is None:
        return out

    delta_rho, g, _rho1 = _physics(physics)
    gamma_n_m = _surface_tension_from_h(delta_rho, g, de_mm, h)
    r0_mm = _r0_mm_from_context(ctx)
    beta = _beta_from_gamma(delta_rho, g, r0_mm, gamma_n_m)
    out.update(
        {
            f"{prefix}_h": h,
            f"{prefix}_surface_tension_mN_m": gamma_n_m * 1000.0,
            f"{prefix}_beta": beta,
        }
    )
    return out


def selected_plane(ctx: Any, profile_mm: np.ndarray, physics: dict[str, Any]) -> dict[str, Any]:
    """Approximate IFT from one selected plane at ``k=1.0``."""
    return _selected_plane_estimate(ctx, profile_mm, physics, k=1.0)


def multi_selected_plane(
    ctx: Any, profile_mm: np.ndarray, physics: dict[str, Any]
) -> dict[str, Any]:
    """Approximate IFT from a median over multiple selected planes."""
    estimates = []
    plane_rows = []
    for k in DEFAULT_SELECTED_PLANES:
        raw = _selected_plane_estimate(ctx, profile_mm, physics, k=k)
        gamma = raw.get("approx_selected_plane_surface_tension_mN_m")
        status = raw.get("approx_selected_plane_status")
        plane_rows.append(
            {
                "k": k,
                "status": status,
                "surface_tension_mN_m": gamma,
                "s": raw.get("approx_selected_plane_s"),
            }
        )
        if status == "ok" and gamma is not None and np.isfinite(float(gamma)):
            estimates.append(float(gamma))

    prefix = "approx_multi_selected_plane"
    out: dict[str, Any] = {
        f"{prefix}_planes": plane_rows,
        f"{prefix}_n": len(estimates),
    }
    if not estimates:
        out[f"{prefix}_status"] = "no_valid_planes"
        return out

    gamma_mn_m = float(np.median(estimates))
    delta_rho, g, _rho1 = _physics(physics)
    r0_mm = _r0_mm_from_context(ctx)
    beta = _beta_from_gamma(delta_rho, g, r0_mm, gamma_mn_m / 1000.0)
    out.update(
        {
            f"{prefix}_status": "ok",
            f"{prefix}_surface_tension_mN_m": gamma_mn_m,
            f"{prefix}_std_mN_m": float(np.std(estimates)),
            f"{prefix}_beta": beta,
        }
    )
    return out


@lru_cache(maxsize=128)
def _volume_lookup(height_over_r0: float) -> tuple[np.ndarray, np.ndarray]:
    h_dim = max(float(height_over_r0), 0.1)
    beta_grid = np.linspace(0.03, 2.5, 90)
    v_values: list[float] = []
    betas: list[float] = []
    for beta in beta_grid:
        model = integrate_young_laplace_profile_mm(
            1.0, float(beta), target_height_mm=h_dim, branch="right", max_step=0.06
        )
        r, z = _profile(model)
        if r.size < 3:
            continue
        v = float(np.pi * trapezoid(r**2, z))
        if np.isfinite(v):
            v_values.append(v)
            betas.append(float(beta))
    v_arr = np.asarray(v_values, dtype=float)
    beta_arr = np.asarray(betas, dtype=float)
    order = np.argsort(v_arr)
    v_arr = v_arr[order]
    beta_arr = beta_arr[order]
    v_unique, idx = np.unique(v_arr, return_index=True)
    return v_unique, beta_arr[idx]


def volume_apex_lookup(ctx: Any, profile_mm: np.ndarray, physics: dict[str, Any]) -> dict[str, Any]:
    """Approximate IFT from volume, height, and apex radius."""
    prefix = "approx_volume_apex"
    r, z = _profile(profile_mm)
    r0_mm = _r0_mm_from_context(ctx)
    if r.size < 3 or r0_mm is None or r0_mm <= 0:
        return {f"{prefix}_status": "missing_profile_or_r0"}

    volume_uL = float(np.pi * trapezoid(r**2, z))
    height_mm = float(np.max(z))
    volume_dim = volume_uL / (r0_mm**3)
    height_dim = height_mm / r0_mm
    v_arr, beta_arr = _volume_lookup(round(height_dim, 4))
    out = {
        f"{prefix}_volume_uL": volume_uL,
        f"{prefix}_height_over_r0": height_dim,
        f"{prefix}_volume_over_r0_cubed": volume_dim,
    }
    if v_arr.size < 3 or volume_dim < float(np.min(v_arr)) or volume_dim > float(np.max(v_arr)):
        out[f"{prefix}_status"] = "outside_lookup_range"
        return out

    beta = float(np.interp(volume_dim, v_arr, beta_arr))
    delta_rho, g, _rho1 = _physics(physics)
    gamma_mn_m = float(surface_tension(delta_rho, g, r0_mm, beta) * 1000.0)
    out.update(
        {
            f"{prefix}_status": "ok",
            f"{prefix}_beta": beta,
            f"{prefix}_surface_tension_mN_m": gamma_mn_m,
        }
    )
    return out


PENDANT_APPROXIMATORS = {
    "selected_plane": selected_plane,
    "multi_selected_plane": multi_selected_plane,
    "volume_apex_lookup": volume_apex_lookup,
}

for _name, _fn in PENDANT_APPROXIMATORS.items():
    register_pendant_approximator(_name, _fn)
