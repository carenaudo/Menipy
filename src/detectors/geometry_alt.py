"""Geometry helpers for the alternative contact angle tab."""

from __future__ import annotations

import numpy as np


from ..physics.contact_geom import (
    line_params,
    contour_line_intersections,
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


def mirror_filter(
    contour: np.ndarray,
    line_pt: np.ndarray,
    line_dir: np.ndarray,
    *,
    keep_above: bool = True,
) -> np.ndarray:
    """Return contour points on one side of the substrate line.

    Parameters
    ----------
    contour:
        Polyline of shape ``(N, 2)``.
    line_pt:
        A point on the substrate line.
    line_dir:
        Direction vector of the substrate line.
    keep_above:
        If ``True`` keep points on the left side of ``line_dir``,
        otherwise keep the right side.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must have shape (N,2)")
    line_pt = np.asarray(line_pt, float)
    line_dir = np.asarray(line_dir, float)
    a = line_pt
    b = line_pt + line_dir
    poly = np.vstack([contour, contour[0]])
    clipped = _clip_halfplane(poly, a, b, keep_left=keep_above)
    if len(clipped) > 0 and np.allclose(clipped[0], clipped[-1]):
        clipped = clipped[:-1]
    return clipped


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
        left, right = contour_line_intersections(contour, a, b, c)
    except ValueError:
        return apex, idx

    dl = (left - line_pt) @ nvec
    dr = (right - line_pt) @ nvec
    if np.isclose(dl, dr, atol=1.0):
        apex = 0.5 * (left + right)
    else:
        apex = left if (dl > dr if mode == "sessile" else dl < dr) else right
    return apex, idx


def split_contour_by_line(
    contour: np.ndarray,
    line_pt: np.ndarray,
    line_dir: np.ndarray,
    *,
    keep_above: bool = True,
) -> np.ndarray:
    """Return the contour segment on one side of the substrate line."""
    return mirror_filter(contour, line_pt, line_dir, keep_above=keep_above)


def _polygon_area(poly: np.ndarray) -> float:
    """Return polygon area using the shoelace formula."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_halfplane(poly: np.ndarray, a: np.ndarray, b: np.ndarray, keep_left: bool) -> np.ndarray:
    """Clip ``poly`` with the half-plane defined by line ``a``-``b``."""
    res: list[np.ndarray] = []
    n = len(poly)
    ab = b - a
    for i in range(n):
        p = poly[i]
        q = poly[(i + 1) % n]
        cp_p = np.cross(ab, p - a)
        cp_q = np.cross(ab, q - a)
        in_p = cp_p >= 0 if keep_left else cp_p <= 0
        in_q = cp_q >= 0 if keep_left else cp_q <= 0
        if in_p:
            res.append(p)
        if in_p ^ in_q:
            t = cp_p / (cp_p - cp_q)
            res.append(p + t * (q - p))
    return np.array(res)


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


def geom_metrics_alt(
    substrate_poly: np.ndarray,
    contour_px: np.ndarray,
    px_per_mm: float,
    *,
    keep_above: bool | None = None,
) -> dict:
    """Return geometric metrics relative to a substrate polyline."""
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")

    line_pt = substrate_poly[0]
    line_dir = substrate_poly[-1] - substrate_poly[0]

    p1, p2 = find_substrate_intersections(contour_px, line_pt, line_dir)
    contact_seg = trim_poly_between(substrate_poly, p1, p2)

    if keep_above is None:
        cont_above = split_contour_by_line(contour_px, line_pt, line_dir, keep_above=True)
        poly_above = np.vstack([cont_above, contact_seg[::-1]])
        area_above = _polygon_area(poly_above) if len(poly_above) >= 3 else 0.0

        cont_below = split_contour_by_line(contour_px, line_pt, line_dir, keep_above=False)
        poly_below = np.vstack([cont_below, contact_seg[::-1]])
        area_below = _polygon_area(poly_below) if len(poly_below) >= 3 else 0.0

        keep_above = area_above >= area_below
        droplet_contour = cont_above if keep_above else cont_below
        droplet_poly = poly_above if keep_above else poly_below
    else:
        droplet_contour = split_contour_by_line(
            contour_px, line_pt, line_dir, keep_above=keep_above
        )
        droplet_poly = np.vstack([droplet_contour, contact_seg[::-1]])

    mode = "sessile" if keep_above else "pendant"
    apex_px, _ = apex_point(droplet_contour, line_pt, line_dir, mode)

    a, b, c = line_params(tuple(p1), tuple(p2))
    h_px = abs(a * apex_px[0] + b * apex_px[1] + c)
    w_px = np.linalg.norm(p2 - p1)

    w_mm = w_px / px_per_mm
    rb_mm = w_mm / 2.0
    h_mm = h_px / px_per_mm

    _, foot = project_pts_onto_poly(np.array([apex_px]), substrate_poly)
    ratio = symmetry_area_ratio(droplet_poly, apex_px, foot[0])

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
        "symmetry_ratio": float(ratio),
        "apex": (int(round(apex_px[0])), int(round(apex_px[1]))),
    }

