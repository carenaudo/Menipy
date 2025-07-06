"""Geometry helpers for processing module."""

from __future__ import annotations

import numpy as np


def clip_line_to_roi(
    point: np.ndarray, direction: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return line intersections with left and right ROI borders.

    Parameters
    ----------
    point:
        A point on the line ``(x, y)``.
    direction:
        Direction vector ``(dx, dy)``.
    width:
        Width of the ROI in pixels. The ROI x-range is ``[0, width - 1]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The left and right intersection points ``(x, y)``.
    """
    point = np.asarray(point, float)
    direction = np.asarray(direction, float)
    if abs(direction[0]) < 1e-6:
        # nearly vertical -- clamp to avoid division by zero
        direction[0] = np.sign(direction[0]) or 1.0
    m = direction[1] / direction[0]
    b = point[1] - m * point[0]
    x1 = 0.0
    y1 = m * x1 + b
    x2 = float(width - 1)
    y2 = m * x2 + b
    return np.array([x1, y1]), np.array([x2, y2])
