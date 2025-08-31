"""Geometry helpers for contact angle analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def line_params(p1_px: tuple[float, float], p2_px: tuple[float, float]) -> tuple[float, float, float]:
    """Return (a, b, c) for line ax + by + c = 0 normalised."""
    x1, y1 = p1_px
    x2, y2 = p2_px
    a, b = y1 - y2, x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)
    if norm == 0:
        return 0.0, 0.0, 0.0
    return a / norm, b / norm, c / norm


def contour_line_intersections(
    contour_px: np.ndarray, a: float, b: float, c: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two contour-line intersection points (left, right)."""
    d = a * contour_px[:, 0] + b * contour_px[:, 1] + c
    sign_d = np.sign(d)
    idx = np.where(np.diff(sign_d))[0].tolist()
    if sign_d[-1] != sign_d[0]:
        idx.append(len(sign_d) - 1)

    pts: list[np.ndarray] = []
    for i in idx:
        p, q = contour_px[i], contour_px[(i + 1) % len(contour_px)]
        dp = a * p[0] + b * p[1] + c
        dq = a * q[0] + b * q[1] + c
        if dp == dq:
            continue
        t = dp / (dp - dq)
        pts.append(p + t * (q - p))

    if not pts:
        raise ValueError("Line does not intersect contour")

    # sort by projection onto the line to handle any orientation
    tvec = np.array([-b, a], float)
    pts.sort(key=lambda P: float(P.dot(tvec)))
    return np.array(pts[0]), np.array(pts[-1])


def contour_line_intersection_near(
    contour_px: np.ndarray,
    a: float,
    b: float,
    c: float,
    ref: tuple[float, float],
) -> tuple[np.ndarray, int]:
    """Return the contour-line intersection closest to ``ref``.

    Parameters
    ----------
    contour_px:
        Closed contour points ``(x, y)``.
    a, b, c:
        Line parameters such that ``ax + by + c = 0``.
    ref:
        Reference point used to select the nearest intersection.

    Returns
    -------
    (point, index)
        ``point`` is the intersection coordinates and ``index`` is the index of
        the contour segment preceding the intersection.
    """

    d = a * contour_px[:, 0] + b * contour_px[:, 1] + c
    sign = np.sign(d)
    idx = np.where(np.diff(sign))[0].tolist()
    if sign[-1] != sign[0]:
        idx.append(len(sign) - 1)

    if not idx:
        raise ValueError("Line does not intersect contour")

    ref_pt = np.asarray(ref, float)
    best_pt: np.ndarray | None = None
    best_idx = -1
    best_dist = np.inf

    for i in idx:
        p, q = contour_px[i], contour_px[(i + 1) % len(contour_px)]
        dp = a * p[0] + b * p[1] + c
        dq = a * q[0] + b * q[1] + c
        if dp == dq:
            continue
        t = dp / (dp - dq)
        inter = p + t * (q - p)
        dist = float(np.hypot(*(inter - ref_pt)))
        if dist < best_dist:
            best_dist = dist
            best_pt = inter
            best_idx = i

    if best_pt is None:
        raise ValueError("Line does not intersect contour")

    return best_pt, best_idx


def geom_metrics(
    p1_px: tuple[float, float],
    p2_px: tuple[float, float],
    contour_px: np.ndarray,
    apex_idx: int,
    px_per_mm: float,
) -> dict:
    """Return geometric metrics relative to a substrate line.

    The droplet is defined as the contour segment below the line that contains
    the apex point. The returned dictionary includes the trimmed droplet polygon
    used for downstream calculations.
    """
    a, b, c = line_params(p1_px, p2_px)

    # flip sign so the apex lies on the negative side of the line
    if a * contour_px[apex_idx, 0] + b * contour_px[apex_idx, 1] + c > 0:
        a, b, c = -a, -b, -c

    left, right = contour_line_intersections(contour_px, a, b, c)

    def _intersection(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        dp = a * p[0] + b * p[1] + c
        dq = a * q[0] + b * q[1] + c
        t = dp / (dp - dq)
        return p + t * (q - p)

    d = a * contour_px[:, 0] + b * contour_px[:, 1] + c
    n = contour_px.shape[0]

    # walk backwards from apex until crossing the substrate
    i = apex_idx
    while True:
        j = (i - 1) % n
        if d[j] > 0 >= d[i]:
            left_pt = _intersection(contour_px[j], contour_px[i])
            start_idx = i
            break
        i = j
        if i == apex_idx:
            raise ValueError("line does not intersect contour")

    # walk forwards from apex until crossing the substrate
    i = apex_idx
    while True:
        j = (i + 1) % n
        if d[j] > 0 >= d[i]:
            right_pt = _intersection(contour_px[i], contour_px[j])
            end_idx = i
            break
        i = j
        if i == apex_idx:
            raise ValueError("line does not intersect contour")

    # gather contour segment from start_idx..end_idx
    seg_idx: list[int] = []
    k = start_idx
    seg_idx.append(k)
    while k != end_idx:
        k = (k + 1) % n
        seg_idx.append(k)
    sub_contour = contour_px[seg_idx]

    # insert exact intersection points
    sub_contour = np.vstack([left_pt, sub_contour, right_pt])
    droplet_poly = np.vstack([sub_contour, right_pt, left_pt])

    w_px = math.hypot(right_pt[0] - left_pt[0], right_pt[1] - left_pt[1])
    w_mm = w_px / px_per_mm
    rb_mm = w_mm / 2.0
    apex_px = contour_px[apex_idx]
    h_px = abs(a * apex_px[0] + b * apex_px[1] + c)
    h_mm = h_px / px_per_mm

    return {
        "xL_px": float(left_pt[0]),
        "xR_px": float(right_pt[0]),
        "w_mm": float(w_mm),
        "rb_mm": float(rb_mm),
        "h_mm": float(h_mm),
        "droplet_poly": droplet_poly,
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "P1_px": tuple(float(v) for v in left_pt),
        "P2_px": tuple(float(v) for v in right_pt),
    }
