"""Sessile drop analysis functions."""

from __future__ import annotations

import numpy as np

from .commons import compute_drop_metrics, find_apex_index
from ..physics.contact_geom import geom_metrics, line_params


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


def _largest_cluster(points: np.ndarray, eps: float = 3.0) -> np.ndarray:
    """Return the largest cluster from ``points`` using a simple BFS."""
    if len(points) == 0:
        return points
    remaining = list(range(len(points)))
    clusters: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            for idx in list(remaining):
                if np.linalg.norm(points[idx] - points[cluster[0]]) <= eps:
                    remaining.remove(idx)
                    cluster.append(idx)
                    changed = True
        clusters.append(cluster)
    largest = max(clusters, key=len)
    return points[np.array(largest)]


def smooth_contour_segment(
    contour: np.ndarray,
    line: tuple[tuple[float, float], tuple[float, float]],
    side: str,
    *,
    delta: float = 3.0,
    min_cluster: int = 50,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Return a smooth contour segment and contact points for sessile mode."""

    from scipy.interpolate import CubicSpline

    p1, p2 = np.asarray(line[0], float), np.asarray(line[1], float)
    d = p2 - p1
    angle = np.arctan2(d[1], d[0])
    rot = np.array(
        [
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)],
        ]
    )

    cont = (contour - p1) @ rot.T

    a, b, c = line_params((0.0, 0.0), (float(np.hypot(d[0], d[1])), 0.0))
    dist = a * cont[:, 0] + b * cont[:, 1] + c

    mask = dist <= -delta
    cont = cont[mask]
    if len(cont) == 0:
        raise ValueError("no contour points after base filtering")

    center_x = float(cont[:, 0].mean())
    if side == "left":
        cont = cont[cont[:, 0] <= center_x]
    elif side == "right":
        cont = cont[cont[:, 0] >= center_x]

    if len(cont) == 0:
        raise ValueError("no contour points after side filtering")

    cont = _largest_cluster(cont)
    if len(cont) < min_cluster:
        # fall back to all points if cluster too small
        cont = cont

    order = np.argsort(cont[:, 0])
    cont_sorted = cont[order]
    xs, idx = np.unique(cont_sorted[:, 0], return_index=True)
    ys = cont_sorted[idx, 1]
    if len(xs) < 4:
        raise ValueError("too few points for spline")

    spline = CubicSpline(xs, ys)

    xs_sample = np.linspace(xs.min(), xs.max(), 1000)
    ys_sample = spline(xs_sample)
    sign = np.sign(ys_sample)
    idx_root = np.where(np.diff(sign))[0]
    if len(idx_root) == 0:
        raise ValueError("spline does not intersect substrate")

    roots: list[float] = []
    for i in idx_root:
        x0, x1 = xs_sample[i], xs_sample[i + 1]
        y0, y1 = ys_sample[i], ys_sample[i + 1]
        if y1 == y0:
            root_x = x0
        else:
            root_x = x0 - y0 * (x1 - x0) / (y1 - y0)
        roots.append(root_x)
    roots = np.array(roots)
    roots.sort()
    xl = roots[0]
    xr = roots[-1] if len(roots) > 1 else roots[0]

    x_clean = np.linspace(xl, xr, 200)
    y_clean = spline(x_clean)
    clean = np.stack([x_clean, y_clean], axis=1)

    P1 = np.array([xl, 0.0]) @ rot + p1
    P2 = np.array([xr, 0.0]) @ rot + p1
    contour_back = clean @ rot + p1

    return contour_back, tuple(P1), tuple(P2)

