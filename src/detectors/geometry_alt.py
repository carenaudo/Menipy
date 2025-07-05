"""Geometry helpers for the alternative contact angle tab."""

from __future__ import annotations

import numpy as np


from ..physics.contact_geom import line_params

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


def geom_metrics_alt(
    substrate_poly: np.ndarray,
    contour_px: np.ndarray,
    apex_idx: int,
    px_per_mm: float,
) -> dict:
    """Return geometric metrics relative to a substrate polyline."""
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    inter = polyline_contour_intersections(substrate_poly, contour_px)
    if len(inter) < 2:
        raise ValueError("substrate does not intersect contour")
    p1, p2 = inter[0], inter[-1]
    a, b, c = line_params(tuple(p1), tuple(p2))
    sign = side_of_polyline(np.array([contour_px[apex_idx]]), substrate_poly)[0]
    if sign * (a * contour_px[apex_idx, 0] + b * contour_px[apex_idx, 1] + c) > 0:
        a, b, c = -a, -b, -c
    d = a * contour_px[:, 0] + b * contour_px[:, 1] + c
    n = contour_px.shape[0]
    def _intersection(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        dp = a * p[0] + b * p[1] + c
        dq = a * q[0] + b * q[1] + c
        t = dp / (dp - dq)
        return p + t * (q - p)

    i = apex_idx
    while True:
        j = (i - 1) % n
        if sign * d[j] > 0 >= sign * d[i]:
            left_pt = _intersection(contour_px[j], contour_px[i])
            start_idx = i
            break
        i = j
        if i == apex_idx:
            raise ValueError("polyline does not intersect contour")

    i = apex_idx
    while True:
        j = (i + 1) % n
        if sign * d[j] > 0 >= sign * d[i]:
            right_pt = _intersection(contour_px[i], contour_px[j])
            end_idx = i
            break
        i = j
        if i == apex_idx:
            raise ValueError("polyline does not intersect contour")

    seg_idx: list[int] = []
    k = start_idx
    seg_idx.append(k)
    while k != end_idx:
        k = (k + 1) % n
        seg_idx.append(k)
    sub_contour = contour_px[seg_idx]

    sub_contour = np.vstack([left_pt, sub_contour, right_pt])
    contact_seg = trim_poly_between(substrate_poly, p1, p2)
    droplet_poly = np.vstack([sub_contour, contact_seg[::-1]])

    w_px = np.linalg.norm(p2 - p1)
    w_mm = w_px / px_per_mm
    rb_mm = w_mm / 2.0
    apex_px = contour_px[apex_idx]
    _, foot = project_pts_onto_poly(np.array([apex_px]), substrate_poly)
    h_px = np.linalg.norm(apex_px - foot[0])
    h_mm = h_px / px_per_mm

    return {
        "xL_px": float(p1[0]),
        "xR_px": float(p2[0]),
        "w_mm": float(w_mm),
        "rb_mm": float(rb_mm),
        "h_mm": float(h_mm),
        "droplet_poly": droplet_poly,
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "contact_segment": contact_seg,
    }

