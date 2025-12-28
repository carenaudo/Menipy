"""Common, low-level geometric utilities."""

from __future__ import annotations

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
    b = x**2 + y**2
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


def find_contact_points_from_contour(
    contour: np.ndarray, contact_line: tuple, tolerance: float = 20.0
) -> tuple:
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
        candidate_idx = np.argsort(dists)[: max(2, int(0.02 * len(contour)))]

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


def detect_baseline_ransac(
    contour: np.ndarray, threshold: float = 2.0, min_samples: int = 10
) -> tuple[np.ndarray, np.ndarray, float]:
    """Detect baseline using RANSAC for robust line fitting with edge filtering.

    Parameters
    ----------
    contour : np.ndarray
        Array shape (N,2) of contour points (x,y).
    threshold : float
        Maximum distance for inlier classification.
    min_samples : int
        Minimum number of samples to fit initial model.

    Returns
    -------
    p1, p2 : np.ndarray
        Two points defining the baseline.
    confidence : float
        Confidence score (0-1) based on inlier ratio.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")

    # Filter edges: prefer points near the bottom of the contour
    y_min = np.min(contour[:, 1])
    y_max = np.max(contour[:, 1])
    bottom_threshold = y_min + 0.3 * (y_max - y_min)  # Bottom 30%
    candidates = contour[contour[:, 1] >= bottom_threshold]

    if len(candidates) < min_samples:
        candidates = contour  # Fallback to all points

    # RANSAC implementation
    best_model = None
    best_inliers = []
    max_iterations = 100

    for _ in range(max_iterations):
        # Random sample
        sample_idx = np.random.choice(
            len(candidates), size=min(min_samples, len(candidates)), replace=False
        )
        sample = candidates[sample_idx]

        if len(sample) < 2:
            continue

        # Fit line to sample
        if len(sample) == 2:
            p1, p2 = sample[0], sample[1]
        else:
            # Fit line using least squares
            x = sample[:, 0]
            y = sample[:, 1]
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            # Define endpoints
            x_min, x_max = np.min(x), np.max(x)
            p1 = np.array([x_min, m * x_min + c])
            p2 = np.array([x_max, m * x_max + c])

        # Compute distances to all candidates
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            continue
        unit_vec = line_vec / line_len

        # Vector from p1 to points
        vec_to_points = candidates - p1
        # Projection scalar
        proj = np.dot(vec_to_points, unit_vec)
        # Distance to line
        dists = np.abs(np.cross(vec_to_points, unit_vec))

        # Inliers
        inliers = dists <= threshold
        inlier_count = np.sum(inliers)

        if inlier_count > len(best_inliers):
            best_inliers = inliers
            best_model = (p1, p2)

    if best_model is None:
        # Fallback: horizontal line at bottom
        x_min, x_max = np.min(contour[:, 0]), np.max(contour[:, 0])
        y_bottom = np.max(contour[:, 1])
        p1 = np.array([x_min, y_bottom])
        p2 = np.array([x_max, y_bottom])
        confidence = 0.1
    else:
        p1, p2 = best_model
        confidence = len(best_inliers) / len(candidates) if len(candidates) > 0 else 0.0

    return p1, p2, confidence


def refine_apex_curvature(
    contour: np.ndarray, window: int = 5, subpixel_steps: int = 10
) -> tuple[np.ndarray, float]:
    """Refine apex detection using curvature-based subpixel refinement.

    Parameters
    ----------
    contour : np.ndarray
        Array shape (N,2) of contour points (x,y).
    window : int
        Window size for curvature estimation.
    subpixel_steps : int
        Number of subpixel steps for refinement.

    Returns
    -------
    apex : np.ndarray
        Refined apex point (x,y).
    confidence : float
        Confidence score (0-1) based on curvature peak strength.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")

    kappa = curvature_estimates(contour, window=window)

    # Find initial apex candidate (highest curvature point)
    apex_idx = np.argmax(kappa)
    if kappa[apex_idx] == 0:
        # Fallback to lowest y
        apex_idx = np.argmin(contour[:, 1])

    # Subpixel refinement around apex
    start_idx = max(0, apex_idx - window)
    end_idx = min(len(contour), apex_idx + window + 1)
    local_contour = contour[start_idx:end_idx]
    local_kappa = kappa[start_idx:end_idx]

    if len(local_contour) < 3:
        apex = contour[apex_idx]
        confidence = 0.5
    else:
        # Fit quadratic to curvature around peak
        indices = np.arange(len(local_contour))
        try:
            coeffs = np.polyfit(indices, local_kappa, 2)
            # Find maximum of quadratic
            peak_idx = (
                -coeffs[1] / (2 * coeffs[0])
                if coeffs[0] != 0
                else len(local_contour) // 2
            )
            peak_idx = np.clip(peak_idx, 0, len(local_contour) - 1)

            # Interpolate position
            idx_floor = int(np.floor(peak_idx))
            idx_ceil = int(np.ceil(peak_idx))
            if idx_floor == idx_ceil:
                apex = local_contour[idx_floor]
            else:
                frac = peak_idx - idx_floor
                apex = (
                    local_contour[idx_floor] * (1 - frac)
                    + local_contour[idx_ceil] * frac
                )

            # Adjust back to global index
            apex += start_idx

        except np.linalg.LinAlgError:
            apex = contour[apex_idx]

        # Confidence based on curvature peak relative to mean
        mean_kappa = np.mean(kappa)
        peak_kappa = kappa[apex_idx]
        confidence = (
            min(1.0, peak_kappa / (mean_kappa + 1e-6)) if mean_kappa > 0 else 0.5
        )

    return apex, confidence


def estimate_contact_angle_tangent(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    window_px: int = 15,
    method: str = "poly",
) -> tuple[float, float]:
    """Estimate contact angle using tangent method near contact point.

    Parameters
    ----------
    contour : np.ndarray
        Array shape (N,2) of contour points (x,y).
    contact_point : np.ndarray
        Contact point (x,y).
    substrate_line : tuple
        Substrate line ((x1,y1), (x2,y2)).
    window_px : int
        Window size in pixels around contact point.
    method : str
        "poly" for polynomial fit, "arc" for circle fit.

    Returns
    -------
    angle_deg : float
        Contact angle in degrees.
    rmse : float
        Root mean square error of the fit.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")

    # Find points within window of contact point
    distances = np.linalg.norm(contour - contact_point, axis=1)
    mask = distances <= window_px
    local_points = contour[mask]

    if len(local_points) < 5:
        return 90.0, 1.0  # Default to 90 degrees with high error

    # Transform to substrate-aligned coordinates
    p1, p2 = np.array(substrate_line[0], dtype=float), np.array(
        substrate_line[1], dtype=float
    )
    substrate_vec = p2 - p1
    substrate_vec /= np.linalg.norm(substrate_vec)
    perp_vec = np.array([-substrate_vec[1], substrate_vec[0]])

    # Local coordinates: x along substrate, y perpendicular
    local_coords = local_points - p1
    x_local = np.dot(local_coords, substrate_vec)
    y_local = np.dot(local_coords, perp_vec)

    if method == "poly":
        # Fit polynomial to y_local vs x_local
        try:
            coeffs = np.polyfit(x_local, y_local, 2)
            # Derivative at contact point (x=0)
            dy_dx = 2 * coeffs[0] * 0 + coeffs[1]
            angle_rad = np.arctan(dy_dx)
            angle_deg = np.degrees(angle_rad)
            # RMSE
            y_pred = np.polyval(coeffs, x_local)
            rmse = np.sqrt(np.mean((y_local - y_pred) ** 2))
        except np.linalg.LinAlgError:
            angle_deg = 90.0
            rmse = 1.0
    elif method == "arc":
        # Fit circle
        try:
            center, radius = fit_circle(local_points)
            # Tangent at contact point
            vec_to_center = center - contact_point
            dist_to_center = np.linalg.norm(vec_to_center)
            if dist_to_center > 0:
                # Angle between radius vector and substrate normal
                cos_theta = np.dot(vec_to_center, perp_vec) / dist_to_center
                angle_rad = np.arccos(np.clip(cos_theta, -1, 1))
                angle_deg = np.degrees(angle_rad)
            else:
                angle_deg = 90.0
            # RMSE approximation
            distances_to_circle = np.abs(
                np.linalg.norm(local_points - center, axis=1) - radius
            )
            rmse = np.sqrt(np.mean(distances_to_circle**2))
        except:
            angle_deg = 90.0
            rmse = 1.0
    else:
        angle_deg = 90.0
        rmse = 1.0

    return angle_deg, rmse


def estimate_contact_angle_circle_fit(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    window_px: int = 30,
) -> tuple[float, float]:
    """Estimate contact angle using circle fit method.

    Parameters
    ----------
    contour : np.ndarray
        Array shape (N,2) of contour points (x,y).
    contact_point : np.ndarray
        Contact point (x,y).
    substrate_line : tuple
        Substrate line ((x1,y1), (x2,y2)).
    window_px : int
        Window size in pixels.

    Returns
    -------
    angle_deg : float
        Contact angle in degrees.
    rmse : float
        Root mean square error of the fit.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")

    # Find points within window
    distances = np.linalg.norm(contour - contact_point, axis=1)
    mask = distances <= window_px
    local_points = contour[mask]

    if len(local_points) < 5:
        return 90.0, 1.0

    try:
        center, radius = fit_circle(local_points)
        # Vector from center to contact point
        vec_to_contact = contact_point - center
        dist = np.linalg.norm(vec_to_contact)
        if dist > 0:
            # Substrate normal direction
            p1, p2 = np.array(substrate_line[0]), np.array(substrate_line[1])
            substrate_vec = p2 - p1
            perp_vec = np.array([-substrate_vec[1], substrate_vec[0]])
            perp_vec /= np.linalg.norm(perp_vec)

            cos_theta = np.dot(vec_to_contact, perp_vec) / dist
            angle_rad = np.arccos(np.clip(cos_theta, -1, 1))
            angle_deg = np.degrees(angle_rad)
        else:
            angle_deg = 90.0

        # RMSE
        distances_to_circle = np.abs(
            np.linalg.norm(local_points - center, axis=1) - radius
        )
        rmse = np.sqrt(np.mean(distances_to_circle**2))
    except:
        angle_deg = 90.0
        rmse = 1.0

    return angle_deg, rmse


def tangent_angle_at_point(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    window_px: int = 15,
) -> tuple[float, float]:
    """Estimate contact angle at a point using local tangent (polynomial fit).

    Parameters
    ----------
    contour : np.ndarray
        Array shape (N,2) of contour points (x,y).
    contact_point : np.ndarray
        Contact point (x,y).
    substrate_line : tuple
        Substrate line ((x1,y1), (x2,y2)).
    window_px : int
        Window size in pixels around contact point.

    Returns
    -------
    angle_deg : float
        Contact angle in degrees.
    uncertainty : float
        Uncertainty estimate (RMSE of fit).
    """
    return estimate_contact_angle_tangent(
        contour, contact_point, substrate_line, window_px, method="poly"
    )


def circle_fit_angle_at_point(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    window_px: int = 30,
) -> tuple[float, float]:
    """Estimate contact angle at a point using local circle fit.

    Parameters
    ----------
    contour : np.ndarray
        Array shape (N,2) of contour points (x,y).
    contact_point : np.ndarray
        Contact point (x,y).
    substrate_line : tuple
        Substrate line ((x1,y1), (x2,y2)).
    window_px : int
        Window size in pixels around contact point.

    Returns
    -------
    angle_deg : float
        Contact angle in degrees.
    uncertainty : float
        Uncertainty estimate (RMSE of fit).
    """
    return estimate_contact_angle_circle_fit(
        contour, contact_point, substrate_line, window_px
    )
