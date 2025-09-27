# src/adsa/common/solver.py
from __future__ import annotations

from typing import Callable, Iterable, Literal, Optional, Sequence
from menipy.models.fit import FitConfig
import math
import numpy as np

try:
    from scipy.optimize import least_squares  # recommended backend
except Exception:  # pragma: no cover
    least_squares = None  # allow importing without SciPy; fail nicely at runtime


def _residuals_pointwise(obs_xy: np.ndarray, model_xy: np.ndarray) -> np.ndarray:
    """
    Simple pointwise residual: pair points by normalized arc-length.
    Both curves are resampled to the same number of points.
    """
    def _arclen_param(xy):
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        return s / (s[-1] if s[-1] > 0 else 1.0)

    def _resample(xy, u, m=400):
        # resample to m points along arc-length with linear interp
        v = _arclen_param(xy)
        xs = np.interp(u, v, xy[:, 0])
        ys = np.interp(u, v, xy[:, 1])
        return np.column_stack([xs, ys])

    m = min(len(obs_xy), 400)
    u = np.linspace(0.0, 1.0, m)
    obs_r = _resample(obs_xy, u, m=m)
    mod_r = _resample(model_xy, u, m=m)
    diff = obs_r - mod_r
    # stack x and y residuals so LSQ can treat both
    return diff.reshape(-1)


def _residuals_normal_projection(obs_xy: np.ndarray, model_xy: np.ndarray) -> np.ndarray:
    """
    Residual as signed distance along approximate normals of the model curve.
    More geometry-aware but slightly heavier. Uses local tangents for normals.
    """
    # estimate tangents on model
    t = np.gradient(model_xy, axis=0)
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)
    n = np.column_stack([-t[:, 1], t[:, 0]])  # rotate tangents to get normals

    # simple nearest-neighbor from obs -> model indices
    # (KD-tree would be faster; linear scan is fine for 200–800 samples)
    def nn_idx(p):
        d = np.sum((model_xy - p) ** 2, axis=1)
        return int(np.argmin(d))

    res = []
    for p in obs_xy:
        j = nn_idx(p)
        res.append(np.dot((p - model_xy[j]), n[j]))
    return np.asarray(res, dtype=float)


def run(
    ctx,
    *,
    integrator: Callable[[np.ndarray, dict, dict | None], np.ndarray],
    # integrator(params, physics, geometry) -> model_xy (Nx2 in same units as ctx.contour.xy)
    config: FitConfig,
) -> dict:
    """
    Generic nonlinear least-squares fit wrapper.

    - Expects ctx.contour.xy as observed (N,2) coordinates (px or mm).
    - 'integrator' generates a model curve from a parameter vector.
      *Examples*: Young–Laplace profile (pendant/sessile), Rayleigh–Lamb radius(t),
      or Jurin meniscus (converted to comparable coordinates/signals).
    - Returns a normalized dict with params, residual metrics, and solver meta.

    Notes:
      • Uses SciPy's `least_squares` with robust losses for outliers. :contentReference[oaicite:1]{index=1}
      • Young–Laplace fitting against full silhouettes is standard in ADSA. :contentReference[oaicite:2]{index=2}
      • Oscillating-drop frequency/damping ↔ γ, ν via Rayleigh–Lamb. :contentReference[oaicite:3]{index=3}
      • Capillary rise (Jurin) relates h, γ, θ, and tube radius. :contentReference[oaicite:4]{index=4}
    """
    if least_squares is None:
        raise RuntimeError("SciPy is required for fitting. Please install scipy.")

    obs_xy = np.asarray(ctx.contour.xy, dtype=float)
    physics = getattr(ctx, "physics", {}) or {}
    geometry = getattr(ctx, "geometry", {}) or None

    # choose residual function
    if config.distance == "normal_projection":
        residual_fn = _residuals_normal_projection
    else:
        residual_fn = _residuals_pointwise

    weights = None
    if config.weights is not None:
        w = np.asarray(list(config.weights), dtype=float).ravel()
        if w.size == 1:
            weights = w.item()
        else:
            weights = w

    # Build LSQ fun(params) -> residual vector
    def fun(x: np.ndarray) -> np.ndarray:
        model_xy = np.asarray(integrator(x, physics, geometry), dtype=float)
        r = residual_fn(obs_xy, model_xy)
        if weights is None:
            return r
        if np.isscalar(weights):
            return r * float(weights)
        # broadcast per-element weights (clip/expand to match length)
        m = min(len(r), len(weights))
        rr = r.copy()
        rr[:m] *= weights[:m]
        if len(r) > m:
            rr[m:] *= weights[-1]  # last weight for any tail
        return rr

    # Solve
    res = least_squares(
        fun,
        x0=np.asarray(config.x0, dtype=float),
        bounds=(
            np.asarray(config.bounds[0], dtype=float),
            np.asarray(config.bounds[1], dtype=float),
        ),
        loss=config.loss,
        f_scale=config.f_scale,
        max_nfev=config.max_nfev,
        verbose=config.verbose,
        method="trf",
    )

    # Collect outputs
    r_vec = res.fun
    rmse = float(math.sqrt(np.mean(r_vec ** 2))) if r_vec.size else float("nan")
    max_abs = float(np.max(np.abs(r_vec))) if r_vec.size else float("nan")

    fit = {
        "params": res.x.tolist(),
        "param_names": list(config.param_names) if config.param_names else None,
        "residuals": {"rmse": rmse, "max_abs": max_abs, "dof": int(r_vec.size - res.x.size), "r": r_vec.tolist()},
        "solver": {
            "backend": "scipy.least_squares",
            "method": "trf",
            "iterations": int(res.nfev),
            "success": bool(res.success),
            "message": str(res.message),
        },
    }

    # Convenience: if one of the params is gamma, propagate it to a top-level field
    if fit["param_names"] and "gamma_mN_per_m" in fit["param_names"]:
        j = fit["param_names"].index("gamma_mN_per_m")
        fit["gamma_mN_per_m"] = float(fit["params"][j])

    ctx.fit = fit
    return fit

