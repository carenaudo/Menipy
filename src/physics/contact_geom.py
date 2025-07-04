"""Geometry helpers for contact angle analysis."""

from __future__ import annotations

import math
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
    idx = np.where(np.diff(np.sign(d)))[0]
    pts: list[np.ndarray] = []
    for i in idx:
        p, q = contour_px[i], contour_px[i + 1]
        dp = a * p[0] + b * p[1] + c
        dq = a * q[0] + b * q[1] + c
        if dp == dq:
            continue
        t = dp / (dp - dq)
        pts.append(p + t * (q - p))
    if not pts:
        raise ValueError("Line does not intersect contour")
    pts = sorted(pts, key=lambda P: P[0])
    return np.array(pts[0]), np.array(pts[-1])


def geom_metrics(
    p1_px: tuple[float, float],
    p2_px: tuple[float, float],
    contour_px: np.ndarray,
    apex_idx: int,
    px_per_mm: float,
) -> dict:
    """Return geometric metrics relative to a substrate line."""
    a, b, c = line_params(p1_px, p2_px)
    (left, right) = contour_line_intersections(contour_px, a, b, c)
    w_px = math.hypot(right[0] - left[0], right[1] - left[1])
    w_mm = w_px / px_per_mm
    rb_mm = w_mm / 2.0
    apex_px = contour_px[apex_idx]
    h_px = abs(a * apex_px[0] + b * apex_px[1] + c)
    h_mm = h_px / px_per_mm
    return {
        "xL_px": float(left[0]),
        "xR_px": float(right[0]),
        "w_mm": float(w_mm),
        "rb_mm": float(rb_mm),
        "h_mm": float(h_mm),
    }
