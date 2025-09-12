"""Geometry helpers for the alternative contact angle tab."""

from __future__ import annotations

import numpy as np


from menipy.pipelines.sessile.geometry_alt import (
    line_params,
    find_contact_points,
)

def trim_poly_between(poly: np.ndarray, p_start: np.ndarray, p_end: np.ndarray) -> np.ndarray:
    """Return polyline segment between ``p_start`` and ``p_end``.

    The order of points in ``poly`` is preserved. ``p_start`` and ``p_end`` do
    not need to coincide with vertices; the closest vertices are used and the
    first and last element of the returned array are replaced with the exact
    start and end coordinates.
    """
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("poly must have shape (N,2)")
    d_start = np.linalg.norm(poly - p_start, axis=1)
    d_end = np.linalg.norm(poly - p_end, axis=1)
    i_start = int(d_start.argmin())
    i_end = int(d_end.argmin())
    if i_start <= i_end:
        seg = poly[i_start : i_end + 1].copy()
    else:
        seg = np.vstack([poly[i_start:], poly[: i_end + 1]])
    seg[0] = p_start
    seg[-1] = p_end
    return seg


def project_pts_onto_poly(pts: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project ``pts`` onto ``poly`` and return distances and foot points."""
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (M,2)")
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 2:
        raise ValueError("poly must be a polyline of shape (N,2)")
    dists = np.empty(len(pts))
    foots = np.empty_like(pts)
    segs = list(zip(poly[:-1], poly[1:]))
    for k, p in enumerate(pts):
        best_d = np.inf
        best_fp = None
        for a, b in segs:
            v = b - a
            denom = float(v.dot(v))
            if denom == 0:
                continue
            t = np.clip(((p - a).dot(v)) / denom, 0.0, 1.0)
            fp = a + t * v
            d = np.linalg.norm(p - fp)
            if d < best_d:
                best_d = d
                best_fp = fp
        dists[k] = best_d
        foots[k] = best_fp
    return dists, foots


def symmetry_axis(apex: np.ndarray, line_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a point and unit vector defining the symmetry axis."""
    apex = np.asarray(apex, float)
    line_dir = np.asarray(line_dir, float)
    nvec = np.array([-line_dir[1], line_dir[0]], float)
    norm = np.hypot(nvec[0], nvec[1])
    if norm == 0:
        raise ValueError("line_dir cannot be zero")
    nvec /= norm
    return apex, nvec
  
def _segment_intersection(
    a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> np.ndarray | None:
    """Return intersection point of two segments or ``None`` if disjoint."""
    r = a2 - a1
    s = b2 - b1
    denom = r[0] * s[1] - r[1] * s[0]
    if denom == 0:
        return None
    qp = b1 - a1
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return a1 + t * r
    return None


def polyline_contour_intersections(
    poly: np.ndarray, contour: np.ndarray
) -> list[np.ndarray]:
    """Return intersection points between ``poly`` and ``contour``."""
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("poly must have shape (N,2)")
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must have shape (M,2)")
    pts: list[tuple[float, float, float]] = []  # (t along poly, x, y)
    for i in range(len(poly) - 1):
        a1 = poly[i]
        a2 = poly[i + 1]
        for j in range(len(contour)):
            b1 = contour[j]
            b2 = contour[(j + 1) % len(contour)]
            p = _segment_intersection(a1, a2, b1, b2)
            if p is not None:
                t = i + np.linalg.norm(p - a1) / np.linalg.norm(a2 - a1)
                pts.append((t, p[0], p[1]))
    pts.sort(key=lambda v: v[0])
    dedup: list[np.ndarray] = []
    for _, x, y in pts:
        p = np.array([x, y], float)
        if not dedup or np.linalg.norm(dedup[-1] - p) > 1e-6:
            dedup.append(p)
    return dedup


def side_of_polyline(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Return sign of ``pts`` relative to ``poly`` (+1 above, -1 below)."""
    _, foots = project_pts_onto_poly(pts, poly)
    idx = np.maximum(np.minimum(
        np.argmin(np.linalg.norm(poly[:-1] - foots[:, None], axis=2), axis=1),
        len(poly) - 2),
        0)
    tangents = poly[idx + 1] - poly[idx]
    vecs = pts - foots
    cross = tangents[:, 0] * vecs[:, 1] - tangents[:, 1] * vecs[:, 0]
    signs = np.sign(cross)
    signs[signs == 0] = 1
    return signs





def find_substrate_intersections(
    contour: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two intersection points between contour and line."""
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must have shape (N,2)")
    line_pt = np.asarray(line_pt, float)
    line_dir = np.asarray(line_dir, float)
    a, b, c = line_params(tuple(line_pt), tuple(line_pt + line_dir))
    d = a * contour[:, 0] + b * contour[:, 1] + c
    inter: list[np.ndarray] = []
    n = len(contour)
    for i in range(n):
        p = contour[i]
        q = contour[(i + 1) % n]
        dp = d[i]
        dq = d[(i + 1) % n]
        if dp == dq:
            continue
        if dp * dq <= 0:
            t = dp / (dp - dq)
            inter.append(p + t * (q - p))
    if len(inter) < 2:
        raise ValueError("line does not intersect contour")
    inter = np.array(inter)
    tvals = (inter - line_pt) @ line_dir
    order = np.argsort(tvals)
    return inter[order[0]], inter[order[-1]]


def apex_point(
    contour: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray, mode: str
) -> tuple[np.ndarray, int]:
    """Return the droplet apex relative to the substrate line.

    Parameters
    ----------
    contour:
        Contour points ``(x, y)``.
    line_pt:
        A point on the substrate line.
    line_dir:
        Direction vector of the substrate line.
    mode:
        ``"sessile"`` or ``"pendant"`` to select the sign of the normal.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must have shape (N,2)")
    if mode not in {"sessile", "pendant"}:
        raise ValueError("mode must be 'sessile' or 'pendant'")

    line_pt = np.asarray(line_pt, float)
    line_dir = np.asarray(line_dir, float)
    nvec = np.array([-line_dir[1], line_dir[0]], float)
    nvec /= np.hypot(nvec[0], nvec[1])

    dist = (contour - line_pt) @ nvec
    idx = int(np.argmax(dist)) if mode == "sessile" else int(np.argmin(dist))
    apex = contour[idx]

    # refine by averaging symmetric intersections when available
    axis_pt, axis_dir = symmetry_axis(apex, line_dir)
    a, b, c = line_params(tuple(axis_pt), tuple(axis_pt + axis_dir))
    try:
        left, right = find_substrate_intersections(contour, a, b, c)
    except ValueError:
        return apex, idx

    dl = (left - line_pt) @ nvec
    dr = (right - line_pt) @ nvec
    if np.isclose(dl, dr, atol=1.0):
        apex = 0.5 * (left + right)
    else:
        apex = left if (dl > dr if mode == "sessile" else dl < dr) else right
    return apex, idx





def _polygon_area(poly: np.ndarray) -> float:
    """Return polygon area using the shoelace formula."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))





def symmetry_area_ratio(poly: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return fraction of area on the left side of line ``a``-``b``."""
    left = _clip_halfplane(poly, a, b, True)
    right = _clip_halfplane(poly, a, b, False)
    if len(left) < 3 or len(right) < 3:
        return 0.0
    area_left = _polygon_area(left)
    area_right = _polygon_area(right)
    if area_left + area_right == 0:
        return 0.0
    return area_left / (area_left + area_right)




