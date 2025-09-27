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


def point_to_segment_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return the perpendicular distance from point pt to segment AB.

    pt, a, b are (2,) arrays.
    """
    # vector projection clamped to segment
    ap = pt - a
    ab = b - a
    ab2 = ab.dot(ab)
    if ab2 == 0:
        return float(np.linalg.norm(ap))
    t = max(0.0, min(1.0, ap.dot(ab) / ab2))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def curvature_estimates(contour: np.ndarray, window: int = 5) -> np.ndarray:
    """Estimate curvature magnitude at each contour point using finite differences.

    Returns an array of length N with non-negative curvature magnitudes. Uses
    a sliding window of `window` points on each side. Window must be >=2.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")
    N = len(contour)
    if window < 2:
        raise ValueError("window must be >= 2")
    kappa = np.zeros(N, dtype=float)
    # Work on closed contours by wrapping indices
    for i in range(N):
        im = (i - window) % N
        ip = (i + window) % N
        p0 = contour[im]
        p1 = contour[i]
        p2 = contour[ip]
        v1 = p1 - p0
        v2 = p2 - p1
        # curvature magnitude ~ |angle between v1 and v2| / avg segment length
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            kappa[i] = 0.0
            continue
        cosang = np.clip((v1.dot(v2)) / (norm1 * norm2), -1.0, 1.0)
        ang = np.arccos(cosang)
        avg_len = 0.5 * (norm1 + norm2)
        kappa[i] = ang / (avg_len + 1e-12)
    return kappa


def find_contact_points_from_contour(contour: np.ndarray, contact_line: tuple, tolerance: float = 20.0) -> tuple:
    """Find left/right contact points on a droplet contour given a user-drawn contact line.

    Parameters
    ----------
    contour : np.ndarray
        Array shape (N,2) of contour points (x,y).
    contact_line : tuple
        Pair of points (x1,y1), (x2,y2) describing the user-drawn contact line.
    tolerance : float
        Maximum distance (in pixels) from the line to consider points as candidates.

    Returns
    -------
    (left_pt, right_pt)
        The chosen contact points as 1D numpy arrays (x,y). Returns (None, None)
        if not enough evidence.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")

    a = np.array(contact_line[0], dtype=float)
    b = np.array(contact_line[1], dtype=float)

    # Compute distance to the user line for each contour point
    dists = np.array([point_to_segment_distance(p, a, b) for p in contour])

    # Candidate points within tolerance
    candidate_idx = np.where(dists <= tolerance)[0]
    if candidate_idx.size == 0:
        # Expand tolerance heuristically
        candidate_idx = np.argsort(dists)[:max(2, int(0.02 * len(contour)))]

    # Estimate curvature and prefer high-curvature points near the line
    kappa = curvature_estimates(contour, window=5)

    # Score candidates by inverse distance times curvature magnitude
    scores = []
    for idx in candidate_idx:
        score = (1.0 / (dists[idx] + 1e-6)) * (kappa[idx] + 1e-6)
        scores.append((score, idx))
    if not scores:
        return (None, None)

    # Sort candidates by x to separate left/right; use contour x positions
    candidates = sorted(scores, key=lambda s: s[0], reverse=True)

    # If fewer than 2 distinct lateral positions among top candidates, fall back to extremes
    top_indices = [idx for _, idx in candidates[:6]]
    xs = contour[top_indices, 0]
    if np.ptp(xs) < 2.0 and len(contour) >= 2:
        # pick leftmost/rightmost based on projection onto the line perpendicular direction
        proj_axis = np.array([-(b - a)[1], (b - a)[0]])
        proj_vals = (contour[:, :2] - a).dot(proj_axis)
        left_idx = int(np.argmin(proj_vals))
        right_idx = int(np.argmax(proj_vals))
        return (contour[left_idx], contour[right_idx])

    # Choose highest-scoring candidate on the left half and right half relative to contact line midpoint
    mid = 0.5 * (a + b)
    left_cands = [i for _, i in candidates if contour[i, 0] <= mid[0]]
    right_cands = [i for _, i in candidates if contour[i, 0] > mid[0]]

    left_pt = contour[left_cands[0]] if left_cands else None
    right_pt = contour[right_cands[0]] if right_cands else None

    # As fallback, choose best two by score and order them left/right
    if left_pt is None or right_pt is None:
        best_two = [idx for _, idx in candidates[:2]]
        if len(best_two) == 2:
            p0 = contour[best_two[0]]
            p1 = contour[best_two[1]]
            if p0[0] <= p1[0]:
                return (p0, p1)
            else:
                return (p1, p0)

    return (left_pt, right_pt)