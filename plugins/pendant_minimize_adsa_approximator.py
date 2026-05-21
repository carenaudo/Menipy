"""Pendant minimize-based ADSA approximator plugin.

This plugin fits Young-Laplace profile parameters to pendant contour side arcs
(apex->left contact and apex->right contact) using scipy.optimize.minimize.
The droplet-needle bridge is excluded by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree

from menipy.common.registry import register_pendant_approximator
from menipy.models.surface_tension import surface_tension
from menipy.pipelines.pendant.strict_young_laplace import (
    integrate_young_laplace_profile_mm,
)


@dataclass(frozen=True)
class _ArcData:
    left_px: np.ndarray
    right_px: np.ndarray
    apex_xy: np.ndarray
    axis_x_px: float
    px_per_mm: float
    snap_left_px: float
    snap_right_px: float


def _normalize_contour_xy(contour: object) -> np.ndarray:
    xy = np.asarray(contour, dtype=float)
    if xy.ndim == 3 and xy.shape[-1] == 2:
        xy = xy.reshape(-1, 2)
    elif xy.ndim == 2 and xy.shape[1] >= 2:
        xy = xy[:, :2]
    else:
        xy = xy.reshape(-1, 2)

    if xy.shape[0] < 3:
        return np.empty((0, 2), dtype=float)

    # Remove non-finite and near-duplicate consecutive points.
    mask = np.all(np.isfinite(xy), axis=1)
    xy = xy[mask]
    if xy.shape[0] < 3:
        return np.empty((0, 2), dtype=float)

    keep = [0]
    for i in range(1, xy.shape[0]):
        if np.linalg.norm(xy[i] - xy[keep[-1]]) > 0.5:
            keep.append(i)
    xy = xy[np.asarray(keep, dtype=int)]

    if xy.shape[0] >= 2 and np.linalg.norm(xy[0] - xy[-1]) <= 0.5:
        xy = xy[:-1]

    return xy if xy.shape[0] >= 3 else np.empty((0, 2), dtype=float)


def _find_apex_index(xy: np.ndarray) -> int:
    y = xy[:, 1]
    x = xy[:, 0]
    y_max = float(np.max(y))
    plateau = np.where(np.abs(y - y_max) <= 1.0)[0]
    if plateau.size == 0:
        return int(np.argmax(y))
    x_med = float(np.median(x))
    best = plateau[np.argmin(np.abs(x[plateau] - x_med))]
    return int(best)


def _nearest_index(xy: np.ndarray, point_xy: np.ndarray) -> tuple[int, float]:
    d2 = np.sum((xy - point_xy[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return idx, float(np.sqrt(d2[idx]))


def _forward_path(start: int, end: int, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    out = [start]
    i = start
    while i != end:
        i = (i + 1) % n
        out.append(i)
        if len(out) > n + 1:
            break
    return np.asarray(out, dtype=int)


def _backward_path(start: int, end: int, n: int) -> np.ndarray:
    path = _forward_path(end, start, n)
    if path.size == 0:
        return path
    return path[::-1]


def _path_quality(path_idx: np.ndarray, xy: np.ndarray, apex_xy: np.ndarray) -> float:
    if path_idx.size < 5:
        return -1.0
    pts = xy[path_idx]
    radial = np.abs(pts[:, 0] - float(apex_xy[0]))
    k = max(5, int(np.ceil(0.1 * radial.size)))
    k = min(k, radial.size)
    dr = np.diff(radial[:k])
    if dr.size == 0:
        return -1.0
    monotone_ratio = float(np.mean(dr >= -0.5))
    span = float(np.max(radial[:k]) - np.min(radial[:k]))
    return monotone_ratio + 0.1 * span


def _choose_side_path(
    i_apex: int,
    i_contact: int,
    i_other_contact: int,
    xy: np.ndarray,
) -> np.ndarray:
    n = xy.shape[0]
    candidates = [
        _forward_path(i_apex, i_contact, n),
        _backward_path(i_apex, i_contact, n),
    ]

    valid: list[tuple[float, np.ndarray]] = []
    for p in candidates:
        if p.size < 3:
            continue
        if int(i_other_contact) in set(p[1:-1].tolist()):
            continue
        q = _path_quality(p, xy, xy[i_apex])
        valid.append((q, p))

    if not valid:
        return np.array([], dtype=int)

    valid.sort(key=lambda item: item[0], reverse=True)
    return valid[0][1]


def _resample_arc(arc_xy: np.ndarray, n_points: int) -> np.ndarray:
    if arc_xy.shape[0] < 2:
        return arc_xy
    diffs = np.diff(arc_xy, axis=0)
    seg = np.sqrt(np.sum(diffs**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-9:
        return arc_xy[:1]
    t = np.linspace(0.0, s[-1], n_points)
    x = np.interp(t, s, arc_xy[:, 0])
    y = np.interp(t, s, arc_xy[:, 1])
    return np.column_stack([x, y])


def _extract_side_arcs(ctx: Any) -> _ArcData | None:
    contour_obj = getattr(ctx, "contour", None)
    contour_xy = None
    if contour_obj is not None and getattr(contour_obj, "xy", None) is not None:
        contour_xy = contour_obj.xy
    elif getattr(ctx, "detected_contour", None) is not None:
        contour_xy = getattr(ctx, "detected_contour")
    elif getattr(ctx, "drop_contour", None) is not None:
        contour_xy = getattr(ctx, "drop_contour")

    xy = _normalize_contour_xy(contour_xy)
    if xy.shape[0] < 10:
        return None

    contacts_obj = getattr(ctx, "contact_points", None)
    if contacts_obj is None:
        return None
    try:
        contacts_raw = np.asarray(contacts_obj, dtype=float)
    except Exception:
        return None
    if contacts_raw.size < 4:
        return None
    contacts2 = np.asarray(contacts_raw.reshape(-1, 2)[:2, :], dtype=float)
    if contacts2.shape[0] < 2:
        return None

    i_apex = _find_apex_index(xy)
    apex_xy = xy[i_apex]

    order = np.argsort(contacts2[:, 0])
    c_left_raw = contacts2[order[0]]
    c_right_raw = contacts2[order[1]]

    bbox = np.ptp(xy, axis=0)
    diag = float(np.hypot(float(bbox[0]), float(bbox[1])))
    snap_max = max(4.0, 0.02 * diag)

    i_left, snap_left = _nearest_index(xy, c_left_raw)
    i_right, snap_right = _nearest_index(xy, c_right_raw)
    if snap_left > snap_max or snap_right > snap_max:
        return None

    if not (xy[i_left, 0] < apex_xy[0] - 1.0 and xy[i_right, 0] > apex_xy[0] + 1.0):
        return None

    left_path_idx = _choose_side_path(i_apex, i_left, i_right, xy)
    right_path_idx = _choose_side_path(i_apex, i_right, i_left, xy)
    if left_path_idx.size < 3 or right_path_idx.size < 3:
        return None

    left_arc = xy[left_path_idx]
    right_arc = xy[right_path_idx]

    y_cut = min(float(xy[i_left, 1]), float(xy[i_right, 1])) - 1.0
    left_arc = left_arc[left_arc[:, 1] >= y_cut]
    right_arc = right_arc[right_arc[:, 1] >= y_cut]
    if left_arc.shape[0] < 3 or right_arc.shape[0] < 3:
        return None

    left_arc = _resample_arc(left_arc, 120)
    right_arc = _resample_arc(right_arc, 120)
    if left_arc.shape[0] < 40 or right_arc.shape[0] < 40:
        return None

    geometry = getattr(ctx, "geometry", None)
    axis_x = float(getattr(geometry, "axis_x", apex_xy[0])) if geometry else float(
        apex_xy[0]
    )
    scale = getattr(ctx, "scale", {}) or {}
    px_per_mm = float(scale.get("px_per_mm", 0.0) or 0.0)
    if px_per_mm <= 0:
        px_per_mm = float(getattr(ctx, "px_per_mm", 0.0) or 0.0)
    if px_per_mm <= 0:
        return None

    return _ArcData(
        left_px=left_arc,
        right_px=right_arc,
        apex_xy=apex_xy,
        axis_x_px=axis_x,
        px_per_mm=px_per_mm,
        snap_left_px=snap_left,
        snap_right_px=snap_right,
    )


def _to_model_mm(points_px: np.ndarray, arc: _ArcData) -> np.ndarray:
    x_mm = (points_px[:, 0] - arc.axis_x_px) / arc.px_per_mm
    z_mm = (arc.apex_xy[1] - points_px[:, 1]) / arc.px_per_mm
    return np.column_stack([x_mm, z_mm])


def _seed_from_ctx(ctx: Any) -> tuple[float, float]:
    results = getattr(ctx, "results", {}) or {}
    fit = getattr(ctx, "fit", {}) or {}

    r0_candidates = [
        results.get("strict_r0_mm"),
        results.get("geometric_r0_mm"),
        results.get("r0_mm"),
        fit.get("strict_r0_mm"),
    ]
    beta_candidates = [
        results.get("strict_beta"),
        results.get("geometric_beta"),
        results.get("beta"),
        fit.get("strict_beta"),
    ]

    r0 = 1.0
    beta = 0.3
    for value in r0_candidates:
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if np.isfinite(v) and v > 0:
            r0 = v
            break

    for value in beta_candidates:
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if np.isfinite(v) and v > 0:
            beta = v
            break

    return float(max(r0, 0.05)), float(np.clip(beta, 0.01, 5.0))


def _objective(
    params: np.ndarray,
    left_mm: np.ndarray,
    right_mm: np.ndarray,
    target_height_mm: float,
) -> float:
    r0_mm, beta, x_off, z_off = params
    if r0_mm <= 0.0 or beta <= 0.0:
        return 1e6

    model_raw = integrate_young_laplace_profile_mm(
        float(r0_mm),
        float(beta),
        target_height_mm=float(target_height_mm),
        branch="right",
    )
    model: np.ndarray = np.asarray(
        model_raw[0] if isinstance(model_raw, tuple) else model_raw,
        dtype=float,
    )
    if model.size == 0 or model.shape[0] < 3:
        return 1e6

    model_r = model + np.array([x_off, z_off], dtype=float)
    model_l = np.column_stack([-model[:, 0], model[:, 1]]) + np.array(
        [x_off, z_off], dtype=float
    )

    tree_l = KDTree(model_l)
    tree_r = KDTree(model_r)
    dl, _ = tree_l.query(left_mm)
    dr, _ = tree_r.query(right_mm)

    # Side-balanced robust mean square.
    def robust_mse(d: np.ndarray) -> float:
        d = np.asarray(d, dtype=float)
        if d.size == 0:
            return 1e6
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med))) + 1e-9
        clip = med + 3.0 * 1.4826 * mad
        d = np.minimum(d, clip)
        return float(np.mean(d**2))

    reg = 1e-4 * (x_off**2 + z_off**2)
    return 0.5 * robust_mse(dl) + 0.5 * robust_mse(dr) + reg


def minimize_adsa(ctx: Any, profile_mm: np.ndarray, physics: dict[str, Any]) -> dict[str, Any]:
    """Fit pendant side arcs with a minimize-based ADSA fallback."""
    prefix = "approx_minimize_adsa"
    arc = _extract_side_arcs(ctx)
    if arc is None:
        return {f"{prefix}_status": "invalid_contact_geometry"}

    left_mm = _to_model_mm(arc.left_px, arc)
    right_mm = _to_model_mm(arc.right_px, arc)

    height_mm = max(float(np.max(left_mm[:, 1])), float(np.max(right_mm[:, 1])))
    width_mm = float(np.max(np.abs(np.concatenate([left_mm[:, 0], right_mm[:, 0]]))))
    r0_seed, beta_seed = _seed_from_ctx(ctx)

    x0 = np.array([r0_seed, beta_seed, 0.0, 0.0], dtype=float)
    bounds = [
        (0.02, max(20.0, 3.0 * max(width_mm, height_mm, r0_seed))),
        (1e-3, 5.0),
        (-max(0.5, 0.2 * width_mm), max(0.5, 0.2 * width_mm)),
        (-max(0.5, 0.2 * height_mm), max(0.5, 0.2 * height_mm)),
    ]

    res = minimize(
        _objective,
        x0=x0,
        args=(left_mm, right_mm, height_mm),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300},
    )

    if (not bool(res.success)) or (not np.all(np.isfinite(res.x))):
        return {
            f"{prefix}_status": "optimization_failed",
            f"{prefix}_solver": {
                "backend": "scipy.minimize",
                "method": "L-BFGS-B",
                "success": bool(res.success),
                "iterations": int(getattr(res, "nit", 0)),
                "message": str(getattr(res, "message", "")),
            },
            f"{prefix}_n_left": int(left_mm.shape[0]),
            f"{prefix}_n_right": int(right_mm.shape[0]),
            f"{prefix}_contact_snap_px_left": float(arc.snap_left_px),
            f"{prefix}_contact_snap_px_right": float(arc.snap_right_px),
            f"{prefix}_arc_quality_status": "invalid",
        }

    r0_mm, beta, x_off, z_off = [float(v) for v in res.x]
    delta_rho = float(physics.get("rho1", 1000.0)) - float(physics.get("rho2", 1.2))
    g = float(physics.get("g", 9.80665))
    gamma_mn_m = float(surface_tension(delta_rho, g, r0_mm, beta) * 1000.0)

    rmse_mm = float(np.sqrt(max(float(res.fun), 0.0)))

    return {
        f"{prefix}_status": "ok",
        f"{prefix}_surface_tension_mN_m": gamma_mn_m,
        f"{prefix}_beta": beta,
        f"{prefix}_r0_mm": r0_mm,
        f"{prefix}_x_offset_mm": x_off,
        f"{prefix}_z_offset_mm": z_off,
        f"{prefix}_rmse_mm": rmse_mm,
        f"{prefix}_solver": {
            "backend": "scipy.minimize",
            "method": "L-BFGS-B",
            "success": bool(res.success),
            "iterations": int(getattr(res, "nit", 0)),
            "message": str(getattr(res, "message", "")),
        },
        f"{prefix}_n_left": int(left_mm.shape[0]),
        f"{prefix}_n_right": int(right_mm.shape[0]),
        f"{prefix}_contact_snap_px_left": float(arc.snap_left_px),
        f"{prefix}_contact_snap_px_right": float(arc.snap_right_px),
        f"{prefix}_arc_quality_status": "ok",
    }


PENDANT_APPROXIMATORS = {"minimize_adsa": minimize_adsa}
register_pendant_approximator("minimize_adsa", minimize_adsa)
