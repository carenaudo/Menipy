"""Substrate line detection utilities."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from skimage.measure import LineModelND, ransac


def clip_line_to_roi(
    point: np.ndarray, direction: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return line intersections with left and right ROI borders."""
    point = np.asarray(point, float)
    direction = np.asarray(direction, float)
    if abs(direction[0]) < 1e-6:
        direction[0] = np.sign(direction[0]) or 1.0
    m = direction[1] / direction[0]
    b = point[1] - m * point[0]
    x1 = 0.0
    y1 = m * x1 + b
    x2 = float(width - 1)
    y2 = m * x2 + b
    return np.array([x1, y1]), np.array([x2, y2])


class SubstrateNotFoundError(RuntimeError):
    """Raised when the substrate line cannot be reliably detected."""


def detect_substrate_line(
    gray_roi: np.ndarray,
    mask: np.ndarray,
    mode: Literal["sessile", "pendant"],
) -> tuple[np.ndarray, np.ndarray]:
    """Return left and right substrate line endpoints in ROI coordinates."""
    if gray_roi.ndim != 2:
        raise ValueError("gray_roi must be grayscale")
    if gray_roi.shape != mask.shape:
        raise ValueError("gray_roi and mask must have the same shape")
    if mode not in {"sessile", "pendant"}:
        raise ValueError("mode must be 'sessile' or 'pendant'")

    blur = cv2.GaussianBlur(gray_roi, (0, 0), 1.5)
    edges = cv2.Canny(blur, 50, 150)

    h, w = gray_roi.shape
    line_segs = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(1, int(0.5 * h)),
        minLineLength=int(0.4 * w),
        maxLineGap=5,
    )
    if line_segs is None or len(line_segs) == 0:
        raise SubstrateNotFoundError("no line segments found")

    band_h = int(0.3 * h)
    if mode == "sessile":
        band_min = h - band_h
        band_max = h
    else:
        band_min = 0
        band_max = band_h

    pts = []
    for x1, y1, x2, y2 in line_segs[:, 0]:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if ang > 90.0:
            ang -= 180.0
        elif ang < -90.0:
            ang += 180.0
        if abs(ang) > 45.0:
            continue
        my = (y1 + y2) / 2.0
        if band_min <= my <= band_max:
            pts.append([x1, y1])
            pts.append([x2, y2])
    if len(pts) < 2:
        raise SubstrateNotFoundError("no segments in band")

    pts = np.array(pts, dtype=float)
    try:
        line_model, inliers = ransac(
            pts,
            LineModelND,
            min_samples=2,
            residual_threshold=4.0,
            max_trials=50,
        )
    except Exception as exc:
        raise SubstrateNotFoundError(str(exc)) from exc

    inlier_ratio = inliers.mean() if inliers.size > 0 else 0.0
    if inlier_ratio < 0.75:
        raise SubstrateNotFoundError("low confidence")

    origin = line_model.params[0]
    direction = line_model.params[1]

    theta = np.degrees(np.arctan2(direction[1], direction[0]))
    if theta > 90.0:
        theta -= 180.0
    elif theta < -90.0:
        theta += 180.0
    if abs(theta) > 45.0:
        raise SubstrateNotFoundError("line orientation invalid")

    p_left, p_right = clip_line_to_roi(origin, direction, w)
    return p_left, p_right
