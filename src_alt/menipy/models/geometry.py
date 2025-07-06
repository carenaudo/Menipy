"""Geometric fitting utilities."""

from typing import Tuple

import numpy as np
from numpy.linalg import lstsq


def fit_circle(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a circle to 2D points using linear least squares.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2) with x, y coordinates.

    Returns
    -------
    center : np.ndarray
        Circle center (x, y).
    radius : float
        Circle radius.
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    c, residuals, _, _ = lstsq(A, b, rcond=None)
    center = c[:2]
    radius = np.sqrt(c[2] + center.dot(center))
    return center, radius


def horizontal_intersections(contour: np.ndarray, y: float) -> np.ndarray:
    """Return x-positions where ``contour`` crosses the horizontal line ``y``."""
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")

    xs = []
    pts1 = contour
    pts2 = np.roll(contour, -1, axis=0)
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        if (y1 - y) * (y2 - y) <= 0 and y1 != y2:
            t = (y - y1) / (y2 - y1)
            xs.append(float(x1 + t * (x2 - x1)))
    return np.array(xs, dtype=float)

