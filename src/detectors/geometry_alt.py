"""Geometry helpers for the alternative contact angle tab."""

from __future__ import annotations

import numpy as np


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


def symmetry_axis(
    apex: np.ndarray | None,
    substrate_poly: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two points defining the symmetry axis."""
    if apex is not None:
        _, foot = project_pts_onto_poly(np.array([apex]), substrate_poly)
        foot = foot[0]
        idx = int(np.linalg.norm(substrate_poly[:-1] - foot, axis=1).argmin())
        tangent = substrate_poly[idx + 1] - substrate_poly[idx]
        normal = np.array([-tangent[1], tangent[0]])
        if np.allclose(normal, 0):
            normal = np.array([0.0, -1.0])
        normal /= np.linalg.norm(normal)
        start = apex
        end = apex + 1000 * normal
        return start, end
    xm = 0.5 * (p1[0] + p2[0])
    start = np.array([xm, (p1[1] + p2[1]) / 2])
    end = start + np.array([0.0, -1000.0])
    return start, end
