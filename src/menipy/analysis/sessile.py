"""Sessile drop analysis functions."""

from __future__ import annotations

import numpy as np

from .commons import compute_drop_metrics, find_apex_index
from ..physics.contact_geom import geom_metrics


def compute_metrics(
    contour: np.ndarray,
    px_per_mm: float,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> dict:
    """Return sessile-drop metrics for ``contour``."""
    if substrate_line is None:
        return compute_drop_metrics(contour, px_per_mm, "contact-angle")

    apex_idx = find_apex_index(contour, "contact-angle")
    geo = geom_metrics(substrate_line[0], substrate_line[1], contour, apex_idx, px_per_mm)
    droplet_poly = geo.pop("droplet_poly")
    metrics = compute_drop_metrics(
        droplet_poly.astype(float),
        px_per_mm,
        "contact-angle",
        substrate_line=substrate_line,
    )
    metrics.update(geo)

    try:
        cp1, cp2 = contact_points_from_spline(contour, substrate_line, delta=0.5)
        metrics["contact_line"] = (
            (int(round(cp1[0])), int(round(cp1[1]))),
            (int(round(cp2[0])), int(round(cp2[1]))),
        )
    except Exception:
        pass

    return metrics

__all__ = ["compute_metrics"]


def contact_points_from_spline(
    contour: np.ndarray,
    line: tuple[tuple[float, float], tuple[float, float]],
    delta: float,
    *,
    smoothing: float = 1.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return sessile-drop contact points via spline extrapolation."""

    from scipy.interpolate import UnivariateSpline

    p1, p2 = np.asarray(line[0], float), np.asarray(line[1], float)
    d = p2 - p1
    angle = np.arctan2(d[1], d[0])

    rot = np.array(
        [
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)],
        ]
    )
    contour_t = (contour - p1) @ rot.T

    mask = np.abs(contour_t[:, 1]) >= delta
    filtered = contour_t[mask]
    if len(filtered) < 4:
        raise ValueError("Too few points after filtering")

    apex_idx = int(np.argmax(filtered[:, 1]))
    x_apex = filtered[apex_idx, 0]

    left = filtered[filtered[:, 0] <= x_apex]
    right = filtered[filtered[:, 0] >= x_apex]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    spl_l = UnivariateSpline(left[:, 1], left[:, 0], s=smoothing)
    spl_r = UnivariateSpline(right[:, 1], right[:, 0], s=smoothing)

    x_l = float(spl_l(0.0))
    x_r = float(spl_r(0.0))

    contact_l = np.array([x_l, 0.0]) @ rot + p1
    contact_r = np.array([x_r, 0.0]) @ rot + p1
    return tuple(contact_l), tuple(contact_r)


__all__.append("contact_points_from_spline")
