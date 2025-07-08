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
    L = float(np.hypot(d[0], d[1]))

    apex_idx = find_apex_index(contour, "contact-angle")
    apex_y = float((contour[apex_idx] - p1) @ rot.T[1])
    dist = contour_t[:, 1]
    if apex_y > 0:
        dist = -dist

    mask = dist <= -delta
    filtered = contour_t[mask]
    if len(filtered) < 4:
        raise ValueError("Too few points after filtering")

    apex_idx = int(np.argmin(filtered[:, 1]))
    x_apex = filtered[apex_idx, 0]

    left = filtered[filtered[:, 0] <= x_apex]
    right = filtered[filtered[:, 0] >= x_apex]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    spl_l = UnivariateSpline(left[:, 1], left[:, 0], s=smoothing)
    spl_r = UnivariateSpline(right[:, 1], right[:, 0], s=smoothing)

    x_l = float(spl_l(0.0))
    x_r = float(spl_r(0.0))

    x_l = max(0.0, min(L, x_l))
    x_r = max(0.0, min(L, x_r))
    if x_l > x_r:
        x_l, x_r = 0.0, L

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


def remove_island_points(
    points: np.ndarray,
    *,
    eps: float = 2.0,
    min_size: int = 5,
) -> np.ndarray:
    """Return ``points`` without small isolated clusters."""

    if len(points) == 0:
        return points
    remaining = list(range(len(points)))
    clusters: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        queue = [seed]
        while queue:
            idx = queue.pop()
            for j in list(remaining):
                if np.linalg.norm(points[j] - points[idx]) <= eps:
                    remaining.remove(j)
                    cluster.append(j)
                    queue.append(j)
        clusters.append(cluster)

    keep_idx: list[int] = []
    for cluster in clusters:
        if len(cluster) >= min_size:
            keep_idx.extend(cluster)

    if not keep_idx:
        return points
    return points[np.array(sorted(keep_idx))]


def find_spline_roots(spline, y0: float) -> np.ndarray:
    """Return the ``x`` positions where ``spline(x) == y0``."""

    xs = np.linspace(spline.x[0], spline.x[-1], 1000)
    ys = spline(xs) - y0
    sign = np.sign(ys)
    idx = np.where(np.diff(sign))[0]
    roots: list[float] = []
    for i in idx:
        x0, x1 = xs[i], xs[i + 1]
        y0_, y1_ = ys[i], ys[i + 1]
        if y1_ == y0_:
            roots.append(x0)
        else:
            roots.append(x0 - y0_ * (x1 - x0) / (y1_ - y0_))
    return np.sort(np.array(roots))


def select_left_and_right_roots(roots: np.ndarray, center_x: float) -> tuple[float, float]:
    """Return the nearest left/right roots around ``center_x``."""

    if len(roots) < 2:
        raise ValueError("need at least two roots")
    left = roots[roots <= center_x]
    right = roots[roots >= center_x]
    if left.size == 0:
        left_x = roots[0]
    else:
        left_x = left.max()
    if right.size == 0:
        right_x = roots[-1]
    else:
        right_x = right.min()
    if left_x >= right_x:
        left_x = roots[0]
        right_x = roots[-1]
    return float(left_x), float(right_x)


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

    mask = dist < -delta
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

    cont = remove_island_points(cont, eps=3.0, min_size=min_cluster)
    if len(cont) == 0:
        raise ValueError("no contour points after clustering")

    order = np.argsort(cont[:, 0])
    cont_sorted = cont[order]
    xs, idx = np.unique(cont_sorted[:, 0], return_index=True)
    ys = cont_sorted[idx, 1]
    if len(xs) < 4:
        raise ValueError("too few points for spline")

    spline = CubicSpline(xs, ys)

    roots = find_spline_roots(spline, 0.0)
    xl, xr = select_left_and_right_roots(roots, center_x)

    x_clean = np.linspace(xl, xr, 200)
    y_clean = spline(x_clean)
    clean = [(x, y) for x, y in zip(x_clean, y_clean) if y < 0.0]
    clean = np.asarray(clean, float)

    P1 = np.array([xl, 0.0]) @ rot + p1
    P2 = np.array([xr, 0.0]) @ rot + p1
    contour_back = clean @ rot + p1

    return contour_back, tuple(P1), tuple(P2)

