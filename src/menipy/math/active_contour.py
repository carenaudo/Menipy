"""Active contour (snake) mathematical engine for droplet analysis.

This module provides a general, first-principles active contour model for image
segmentation, sub-pixel profile refinement, and contact line dynamics.

Theoretical Foundations and Key Literature:
    1. Classical Active Contour Variational Formulation:
       Kass, M., Witkin, A., & Terzopoulos, D. (1988).
       "Snakes: Active contour models."
       International Journal of Computer Vision, 1(4), 321–331.
       DOI: 10.1007/BF00133570

    2. Balloon & Pressure Forces:
       Cohen, L. D. (1991).
       "On active contour models and balloons."
       CVGIP: Image Understanding, 53(2), 211–218.
       DOI: 10.1016/1049-9660(91)90028-N

    3. Parametric B-Spline Active Contours:
       Brigger, P., Hoeg, J., & Unser, M. (2000).
       "B-spline snakes: A flexible tool for parametric contour detection."
       IEEE Transactions on Image Processing, 9(9), 1484–1496.
       DOI: 10.1109/83.862633

    4. Droplet-Specific Substrate-Constrained Snakes (DropSnake):
       Stalder, A. F., Melchior, T., Müller, M., Sage, D., Blu, T., & Unser, M. (2006).
       "Low-bond axisymmetric drop shape analysis for surface tension and contact
       angle measurements of sessile drops."
       Colloids and Surfaces A: Physicochemical and Engineering Aspects, 286(1-3), 92–103.
       DOI: 10.1016/j.colsurfa.2006.03.008

       Stalder, A. F., Kulik, G., Sage, D., Barbieri, L., & Hoffmann, P. (2010).
       "A snake-based approach to accurate determination of both contact points and contact angles."
       Colloids and Surfaces A: Physicochemical and Engineering Aspects, 364(1-3), 72–81.
       DOI: 10.1016/j.colsurfa.2010.04.040

Attribution & Clean-Room License Notice:
    The mathematical equations, variational energy functionals, and finite-difference
    numerical schemes in this module represent established scientific literature.
    This implementation was written from original mathematical first principles in pure
    Python/NumPy/SciPy/OpenCV for Menipy. It contains zero third-party Java source code,
    JAR bytecode, or non-permissive assets.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _cached_shape_inverse(n, coefficients, boundary, builder, invert):
    """Retain at most eight small, read-only solver matrices per process."""
    alpha, beta, gamma = (float.fromhex(value) for value in coefficients)
    matrix = builder(n, alpha, beta, boundary) + gamma * np.eye(n, dtype=float)
    inverse = invert(matrix)
    inverse.setflags(write=False)
    return inverse


def _shape_inverse(n, config, boundary):
    if n > 300:
        # Large standalone snakes must not inflate the temporal tracking cache.
        matrix = build_pentadiagonal_matrix(n, config.alpha, config.beta, boundary)
        return np.linalg.inv(matrix + config.gamma * np.eye(n, dtype=float))
    coefficients = tuple(float(value).hex() for value in (config.alpha, config.beta, config.gamma))
    return _cached_shape_inverse(
        n, coefficients, boundary, build_pentadiagonal_matrix, np.linalg.inv
    )


# -----------------------------------------------------------------------------
# Enums and Configurations
# -----------------------------------------------------------------------------


class SnakeBoundaryCondition(str, Enum):
    """Boundary condition governing the topology and endpoints of the active contour."""

    PERIODIC = "periodic"  # Closed loop (e.g. pendant drops, free bubbles)
    SLIDING_LINE = "sliding_line"  # Open curve whose endpoints slide along a substrate line
    PINNED = "pinned"  # Open curve with fixed endpoint coordinates (e.g. needle cannula)
    FREE = "free"  # Open curve with natural boundary conditions (d²v/ds² = 0, d³v/ds³ = 0)


@dataclass
class ActiveContourConfig:
    """Configuration parameters for the active contour energy minimization."""

    alpha: float = 0.01  # Elasticity (length / tension) penalty weight
    beta: float = 0.1  # Rigidity (bending / smoothness) penalty weight
    gamma: float = 0.01  # Time step viscosity / damping coefficient
    w_line: float = 0.0  # Weight for intensity attraction (negative for dark objects)
    w_edge: float = 1.0  # Weight for gradient magnitude attraction
    w_flux: float = 0.0  # Weight for directional outward normal gradient flux
    w_balloon: float = 0.0  # Weight for balloon (inflation/deflation) normal pressure force
    max_px_move: float = 1.0  # Maximum vertex displacement per iteration (in pixels)
    max_iterations: int = 300  # Maximum optimization iterations
    convergence: float = 0.05  # Convergence threshold on max node movement (pixels)
    resample_interval: int = 10  # Resample equidistant nodes every K iterations (0 to disable)
    gaussian_sigma: float = 2.0  # Sigma for Gaussian image smoothing


@dataclass
class ActiveContourResult:
    """Result returned by active contour evolution."""

    xy: np.ndarray  # Converged (N, 2) array of coordinates (x, y)
    tangents: np.ndarray  # Unit tangent vectors (N, 2)
    normals: np.ndarray  # Unit outward normal vectors (N, 2)
    curvatures: np.ndarray  # Signed local curvature kappa (N,)
    converged: bool  # Whether evolution met convergence tolerance
    iterations: int  # Number of iterations executed
    energy: float  # Final estimated average energy
    boundary_condition: SnakeBoundaryCondition


@dataclass
class BSplineSnakeResult:
    """Result of continuous parametric B-spline fitting to a droplet contour."""

    xy: np.ndarray  # Evaluated (M, 2) points along the B-spline curve
    tangents: np.ndarray  # Unit tangent vectors (M, 2)
    normals: np.ndarray  # Unit normal vectors (M, 2)
    curvatures: np.ndarray  # Local curvature kappa (M,)
    contact_angles_deg: tuple[float, float] | None  # (theta_left, theta_right) in degrees
    tck: tuple[Any, ...]  # SciPy B-spline representation (t, c, k)


# -----------------------------------------------------------------------------
# Line Projection & Geometry Utilities
# -----------------------------------------------------------------------------


def project_point_to_line(
    point: np.ndarray | tuple[float, float] | list[float] | Sequence[float],
    line: tuple[tuple[float, float], tuple[float, float]] | tuple[float, float, float] | float,
) -> np.ndarray:
    """Project a 2D point orthogonally onto a line.

    The line may be specified as:
        1. Two points defining the line: ((x1, y1), (x2, y2))
        2. Standard line coefficients (A, B, C) for A*x + B*y + C = 0
        3. A scalar float y0 for a horizontal line y = y0

    Args:
        point: Coordinates (x, y) of the point to project.
        line: Line definition (points tuple, coefficients tuple, or scalar horizontal Y).

    Returns:
        Projected point as a 2-element numpy array [x_proj, y_proj].
    """
    p = np.asarray(point, dtype=float).ravel()
    px, py = p[0], p[1]

    # Case 1: Scalar horizontal line y = y0
    if isinstance(line, (int, float)):
        return np.array([px, float(line)], dtype=float)

    # Case 2: Tuple of two points ((x1, y1), (x2, y2))
    if len(line) == 2 and isinstance(line[0], (tuple, list, np.ndarray)):
        p1 = np.asarray(line[0], dtype=float)
        p2 = np.asarray(line[1], dtype=float)
        v = p2 - p1
        v_norm_sq = float(np.dot(v, v))
        if v_norm_sq < 1e-12:
            return p1.copy()
        t = float(np.dot(p - p1, v)) / v_norm_sq
        return p1 + t * v

    # Case 3: Line coefficients (A, B, C) where A*x + B*y + C = 0
    if len(line) == 3:
        A, B, C = float(line[0]), float(line[1]), float(line[2])
        denom = A * A + B * B
        if denom < 1e-12:
            return np.array([px, py], dtype=float)
        dist = (A * px + B * py + C) / denom
        return np.array([px - A * dist, py - B * dist], dtype=float)

    raise ValueError(f"Unsupported line representation: {line}")


def compute_substrate_angle(
    line: tuple[tuple[float, float], tuple[float, float]] | tuple[float, float, float] | float,
) -> float:
    """Compute the angle of the substrate line relative to horizontal in degrees.

    Returns an angle in [-90, 90] degrees.
    """
    if isinstance(line, (int, float)):
        return 0.0

    if len(line) == 2 and isinstance(line[0], (tuple, list, np.ndarray)):
        p1 = np.asarray(line[0], dtype=float)
        p2 = np.asarray(line[1], dtype=float)
        # Ensure left-to-right orientation
        if p2[0] < p1[0]:
            p1, p2 = p2, p1
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return float(np.degrees(np.arctan2(dy, dx)))

    if len(line) == 3:
        A, B = float(line[0]), float(line[1])
        # Normal is (A, B), tangent is (-B, A) or (B, -A)
        # For A*x + B*y + C = 0, dy/dx = -A/B
        if abs(B) < 1e-9:
            return 90.0
        return float(np.degrees(np.arctan(-A / B)))

    return 0.0


def get_substrate_direction_vector(
    line: tuple[tuple[float, float], tuple[float, float]] | tuple[float, float, float] | float | None,
    default_left_to_right: np.ndarray | None = None,
) -> np.ndarray:
    """Return a unit direction vector (dx, dy) pointing along the substrate from left to right."""
    if line is None:
        if default_left_to_right is not None and len(default_left_to_right) >= 2:
            v = default_left_to_right[-1] - default_left_to_right[0]
            norm = float(np.hypot(v[0], v[1]))
            return v / (norm if norm > 1e-9 else 1.0)
        return np.array([1.0, 0.0], dtype=float)

    if isinstance(line, (int, float)):
        return np.array([1.0, 0.0], dtype=float)

    if len(line) == 2 and isinstance(line[0], (tuple, list, np.ndarray)):
        p1 = np.asarray(line[0], dtype=float)
        p2 = np.asarray(line[1], dtype=float)
        if p2[0] < p1[0]:
            p1, p2 = p2, p1
        v = p2 - p1
        norm = float(np.hypot(v[0], v[1]))
        return v / (norm if norm > 1e-9 else 1.0)

    if len(line) == 3:
        A, B = float(line[0]), float(line[1])
        v = np.array([-B, A], dtype=float)
        if v[0] < 0:
            v = -v
        norm = float(np.hypot(v[0], v[1]))
        return v / (norm if norm > 1e-9 else 1.0)

    return np.array([1.0, 0.0], dtype=float)


# -----------------------------------------------------------------------------
# Shape Matrix Construction
# -----------------------------------------------------------------------------


def build_pentadiagonal_matrix(
    n: int,
    alpha: float,
    beta: float,
    bc: SnakeBoundaryCondition,
) -> np.ndarray:
    """Construct the Kass et al. (1988) pentadiagonal internal shape matrix.

    Minimizes internal elasticity (alpha * |v'|²) and rigidity (beta * |v''|²)
    subject to the chosen boundary condition.

    Args:
        n: Number of contour vertices (must be >= 4).
        alpha: Elasticity coefficient (higher values encourage shrinking).
        beta: Rigidity coefficient (higher values enforce smoothness).
        bc: Snake boundary condition (PERIODIC, PINNED, FREE, or SLIDING_LINE).

    Returns:
        (n, n) float matrix A representing the internal variational forces.
    """
    if n < 4:
        raise ValueError(f"Snake requires at least 4 points, got n={n}")

    if bc == SnakeBoundaryCondition.PERIODIC:
        eye_n = np.eye(n, dtype=float)
        # Second derivative (central difference)
        a = (
            np.roll(eye_n, -1, axis=0)
            + np.roll(eye_n, 1, axis=0)
            - 2.0 * eye_n
        )
        # Fourth derivative (central difference)
        b = (
            np.roll(eye_n, -2, axis=0)
            + np.roll(eye_n, 2, axis=0)
            - 4.0 * np.roll(eye_n, -1, axis=0)
            - 4.0 * np.roll(eye_n, 1, axis=0)
            + 6.0 * eye_n
        )
        return -alpha * a + beta * b

    # Open curve formulations
    a = np.zeros((n, n), dtype=float)
    b = np.zeros((n, n), dtype=float)

    # Interior nodes
    for i in range(1, n - 1):
        a[i, i - 1 : i + 2] = [1.0, -2.0, 1.0]

    for i in range(2, n - 2):
        b[i, i - 2 : i + 3] = [1.0, -4.0, 6.0, -4.0, 1.0]

    if bc == SnakeBoundaryCondition.PINNED:
        # Pinned endpoints: row 0 and row n-1 have zero internal force
        # Near endpoints, use second-order difference
        if n > 2:
            b[1, :3] = [1.0, -2.0, 1.0]
            b[n - 2, n - 3 :] = [1.0, -2.0, 1.0]
    else:
        # Natural / sliding boundary conditions (FREE or SLIDING_LINE)
        # Natural BC sets second and fourth derivatives to zero at boundaries
        a[0, :2] = [-1.0, 1.0]
        a[n - 1, n - 2 :] = [1.0, -1.0]

        b[0, :3] = [1.0, -2.0, 1.0]
        b[1, :4] = [-1.0, 3.0, -3.0, 1.0]
        b[n - 2, n - 4 :] = [-1.0, 3.0, -3.0, 1.0]
        b[n - 1, n - 3 :] = [1.0, -2.0, 1.0]

    return -alpha * a + beta * b


# -----------------------------------------------------------------------------
# Arc-Length Resampling & Normal Vector Computation
# -----------------------------------------------------------------------------


def compute_contour_normals(
    xy: np.ndarray,
    closed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unit tangent vectors, unit outward normals, and local curvatures.

    For an open drop silhouette ordered from left-contact to right-contact,
    the outward normal points away from the drop interior (into the background).

    Args:
        xy: (N, 2) array of coordinates.
        closed: Whether the contour is a closed loop.

    Returns:
        tangents: (N, 2) unit tangent vectors.
        normals: (N, 2) unit outward normal vectors.
        curvatures: (N,) local signed curvatures.
    """
    n = len(xy)
    if n < 3:
        zeros_2d = np.zeros((n, 2), dtype=float)
        return zeros_2d, zeros_2d, np.zeros(n, dtype=float)

    tangents, normals, lengths = _contour_directions(xy, closed)
    tx, ty = tangents[:, 0], tangents[:, 1]

    # Arc-length step ds between adjacent points
    ds = lengths / 2.0
    ds[ds < 1e-9] = 1.0

    if closed:
        prev_t = np.roll(tangents, 1, axis=0)
        next_t = np.roll(tangents, -1, axis=0)
    else:
        prev_t = np.vstack([tangents[0], tangents[:-1]])
        next_t = np.vstack([tangents[1:], tangents[-1]])

    dtx = (next_t[:, 0] - prev_t[:, 0]) / (2.0 * ds)
    dty = (next_t[:, 1] - prev_t[:, 1]) / (2.0 * ds)
    curvatures = dtx * ty - dty * tx

    return tangents, normals, curvatures


def _contour_directions(
    xy: np.ndarray, closed: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central-difference directions for contours with at least three vertices."""
    if closed:
        prev_xy = np.roll(xy, 1, axis=0)
        next_xy = np.roll(xy, -1, axis=0)
    else:
        prev_xy = np.vstack([xy[0], xy[:-1]])
        next_xy = np.vstack([xy[1:], xy[-1]])
    dx = next_xy[:, 0] - prev_xy[:, 0]
    dy = next_xy[:, 1] - prev_xy[:, 1]
    lengths = np.hypot(dx, dy)
    lengths[lengths < 1e-9] = 1.0
    tx = dx / lengths
    ty = dy / lengths
    return np.column_stack([tx, ty]), np.column_stack([ty, -tx]), lengths


def _evolution_normals(
    xy: np.ndarray, closed: bool, config: ActiveContourConfig | None = None
) -> np.ndarray | None:
    """Skip curvature until the final output; evolution only consumes normals."""
    if config is not None and config.w_flux == 0.0 and config.w_balloon == 0.0:
        return None
    return _contour_directions(xy, closed)[1]


def resample_contour_arclength(
    xy: np.ndarray,
    n_points: int | None = None,
    closed: bool = False,
    preserve_endpoints: bool = True,
) -> np.ndarray:
    """Resample contour vertices to ensure uniform equidistant arc-length spacing.

    Prevents node clustering at sharp corners or starvation along flat edges.

    Args:
        xy: (N, 2) array of contour points.
        n_points: Number of points in output contour (defaults to len(xy)).
        closed: Whether the contour is closed.
        preserve_endpoints: If True and open, strictly preserves xy[0] and xy[-1].

    Returns:
        (n_points, 2) resampled contour array.
    """
    if len(xy) < 3:
        return xy.copy()

    if n_points is None:
        n_points = len(xy)

    if closed:
        pts = np.vstack([xy, xy[0]])
    else:
        pts = xy

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_dist[-1]

    if total_len < 1e-9:
        return np.repeat(xy[:1], n_points, axis=0)

    if closed:
        target_s = np.linspace(0.0, total_len, n_points + 1)[:-1]
    else:
        target_s = np.linspace(0.0, total_len, n_points)

    rx = np.interp(target_s, cum_dist, pts[:, 0])
    ry = np.interp(target_s, cum_dist, pts[:, 1])
    resampled = np.column_stack([rx, ry])

    if not closed and preserve_endpoints:
        resampled[0] = xy[0]
        resampled[-1] = xy[-1]

    return resampled


# -----------------------------------------------------------------------------
# External Image Force Field Computation
# -----------------------------------------------------------------------------


def precompute_image_gradients(
    image: np.ndarray, config: ActiveContourConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute static spatial image gradient fields for active contour evolution.

    Args:
        image: Grayscale image (2D uint8 or float array).
        config: ActiveContourConfig containing gaussian_sigma and edge weights.

    Returns:
        (egx, egy, gx, gy) derivative arrays where:
            egx, egy: Spatial derivatives of normalized gradient magnitude (for w_edge).
            gx, gy: Spatial derivatives of smoothed image intensity (for w_line / w_flux).
    """
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0

    # Gaussian smoothing to widen the capture basin
    if config.gaussian_sigma > 0.0:
        ksize = int(2 * np.ceil(2.5 * config.gaussian_sigma) + 1)
        smooth = cv2.GaussianBlur(
            img, (ksize, ksize), sigmaX=config.gaussian_sigma, sigmaY=config.gaussian_sigma
        )
    else:
        smooth = img

    # Spatial image gradients
    gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    gmax = grad_mag.max()
    if gmax > 1e-9:
        grad_mag /= gmax

    # Gradient of gradient magnitude (attracts snake to edges)
    egx = cv2.Sobel(grad_mag, cv2.CV_64F, 1, 0, ksize=3)
    egy = cv2.Sobel(grad_mag, cv2.CV_64F, 0, 1, ksize=3)

    return egx, egy, gx, gy


def compute_external_forces(
    image: np.ndarray,
    xy: np.ndarray,
    normals: np.ndarray | None,
    config: ActiveContourConfig,
    precomputed_gradients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Compute external potential force vectors at each contour vertex.

    Calculates:
        - Gradient magnitude edge attraction: -grad(|grad(I)|)
        - Intensity line attraction: grad(I)
        - Gradient normal flux: (grad(I) . n) * n
        - Balloon normal expansion/contraction: w_balloon * n

    Args:
        image: Grayscale image (2D uint8 or float array).
        xy: (N, 2) array of vertex coordinates (x, y).
        normals: (N, 2) unit normal vectors; may be None when both normal-force
            weights (flux and balloon) are zero.
        config: ActiveContourConfig containing force weights and smoothing sigma.
        precomputed_gradients: Optional precomputed (egx, egy, gx, gy) tuple from
            precompute_image_gradients to avoid redundant recomputations in loops.

    Returns:
        (N, 2) external force vectors (fx, fy) at each vertex.
    """
    if normals is None and (config.w_flux != 0.0 or config.w_balloon != 0.0):
        raise ValueError("Normals are required for flux and balloon forces")
    h, w = image.shape[:2]

    if precomputed_gradients is not None:
        egx, egy, gx, gy = precomputed_gradients
    else:
        egx, egy, gx, gy = precompute_image_gradients(image, config)

    # Sub-pixel bilinear sampling at contour vertices
    px = np.clip(xy[:, 0], 0.0, w - 1.0)
    py = np.clip(xy[:, 1], 0.0, h - 1.0)
    coords = [py, px]

    fx = np.zeros(len(xy), dtype=float)
    fy = np.zeros(len(xy), dtype=float)

    if config.w_edge != 0.0:
        fx += config.w_edge * map_coordinates(egx, coords, order=1, mode="nearest")
        fy += config.w_edge * map_coordinates(egy, coords, order=1, mode="nearest")

    # Line and normal-flux forces use the same image-gradient samples.
    if config.w_line != 0.0 or config.w_flux != 0.0:
        samp_gx = map_coordinates(gx, coords, order=1, mode="nearest")
        samp_gy = map_coordinates(gy, coords, order=1, mode="nearest")

    if config.w_line != 0.0:
        fx += config.w_line * samp_gx
        fy += config.w_line * samp_gy

    if config.w_flux != 0.0:
        flux = samp_gx * normals[:, 0] + samp_gy * normals[:, 1]
        fx += config.w_flux * flux * normals[:, 0]
        fy += config.w_flux * flux * normals[:, 1]

    if config.w_balloon != 0.0:
        fx += config.w_balloon * normals[:, 0]
        fy += config.w_balloon * normals[:, 1]

    return np.column_stack([fx, fy])


# -----------------------------------------------------------------------------
# Main Variational Active Contour Evolution Solver
# -----------------------------------------------------------------------------


def evolve_active_contour(
    image: np.ndarray,
    init_xy: np.ndarray,
    config: ActiveContourConfig | None = None,
    boundary_condition: SnakeBoundaryCondition = SnakeBoundaryCondition.PERIODIC,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | tuple[float, float, float] | float | None = None,
    pinned_endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> ActiveContourResult:
    """Evolve an active contour (snake) on an image using semi-implicit Euler integration.

    Supports:
        - PERIODIC closed loops (pendant drops, bubbles).
        - SLIDING_LINE open curves whose endpoints slide along a substrate baseline.
        - PINNED open curves with fixed boundary endpoints (e.g. needle tips).
        - FREE open curves with unconstrained natural ends.

    Args:
        image: Grayscale image (2D ndarray).
        init_xy: (N, 2) initial contour coordinates.
        config: Optimization configuration parameters.
        boundary_condition: Snake boundary condition enum.
        substrate_line: Baseline definition for SLIDING_LINE condition.
        pinned_endpoints: ((x0, y0), (x1, y1)) coordinates for PINNED condition.

    Returns:
        ActiveContourResult containing converged coordinates, tangents, normals, and diagnostics.
    """
    if config is None:
        config = ActiveContourConfig()

    xy = np.asarray(init_xy, dtype=float).copy()
    n = len(xy)
    if n < 4:
        raise ValueError(f"Active contour requires at least 4 nodes, got {n}")

    closed = (boundary_condition == SnakeBoundaryCondition.PERIODIC)

    # Initial boundary projection
    if boundary_condition == SnakeBoundaryCondition.SLIDING_LINE and substrate_line is not None:
        xy[0] = project_point_to_line(xy[0], substrate_line)
        xy[-1] = project_point_to_line(xy[-1], substrate_line)
    elif boundary_condition == SnakeBoundaryCondition.PINNED and pinned_endpoints is not None:
        xy[0] = np.asarray(pinned_endpoints[0], dtype=float)
        xy[-1] = np.asarray(pinned_endpoints[1], dtype=float)

    # Pre-build shape matrix A and pre-invert (A + gamma * I)
    inv_M = _shape_inverse(n, config, boundary_condition)

    # Precompute static spatial image gradient fields once
    gradients = precompute_image_gradients(image, config)

    h, w = image.shape[:2]
    converged = False
    iterations_run = 0

    x = xy[:, 0].copy()
    y = xy[:, 1].copy()

    # Track recent states to detect convergence even in presence of micro-oscillations
    history_len = 10
    hist_x = np.zeros((history_len, n), dtype=float)
    hist_y = np.zeros((history_len, n), dtype=float)

    for it in range(config.max_iterations):
        iterations_run = it + 1
        curr_xy = np.column_stack([x, y])

        # 1. Compute geometry (tangents, normals)
        normals = _evolution_normals(curr_xy, closed, config)

        # 2. Compute external forces using precomputed gradient fields
        f_ext = compute_external_forces(
            image, curr_xy, normals, config, precomputed_gradients=gradients
        )
        fx, fy = f_ext[:, 0], f_ext[:, 1]

        # 3. Boundary force damping
        if boundary_condition == SnakeBoundaryCondition.PINNED:
            fx[0] = fy[0] = fx[-1] = fy[-1] = 0.0

        # 4. Semi-implicit Euler step
        x_new = inv_M @ (config.gamma * x + fx)
        y_new = inv_M @ (config.gamma * y + fy)

        # 5. Cap maximum vertex movement per iteration using tanh
        dx = config.max_px_move * np.tanh(x_new - x)
        dy = config.max_px_move * np.tanh(y_new - y)

        if boundary_condition == SnakeBoundaryCondition.PINNED:
            dx[0] = dy[0] = dx[-1] = dy[-1] = 0.0

        x += dx
        y += dy

        # 6. Apply boundary constraint projection
        if boundary_condition == SnakeBoundaryCondition.SLIDING_LINE and substrate_line is not None:
            p0_proj = project_point_to_line([x[0], y[0]], substrate_line)
            p1_proj = project_point_to_line([x[-1], y[-1]], substrate_line)
            x[0], y[0] = p0_proj[0], p0_proj[1]
            x[-1], y[-1] = p1_proj[0], p1_proj[1]
        elif boundary_condition == SnakeBoundaryCondition.PINNED and pinned_endpoints is not None:
            x[0], y[0] = pinned_endpoints[0][0], pinned_endpoints[0][1]
            x[-1], y[-1] = pinned_endpoints[1][0], pinned_endpoints[1][1]

        # 7. Clamp within image domain
        x = np.clip(x, 0.0, w - 1.0)
        y = np.clip(y, 0.0, h - 1.0)

        # 8. Periodic arc-length resampling to prevent node tangling
        if config.resample_interval > 0 and (it + 1) % config.resample_interval == 0:
            resampled = resample_contour_arclength(
                np.column_stack([x, y]),
                n_points=n,
                closed=closed,
                preserve_endpoints=True,
            )
            x = resampled[:, 0]
            y = resampled[:, 1]

        # 9. Check convergence using sliding window
        max_disp = float(np.max(np.hypot(dx, dy)))
        if max_disp < config.convergence:
            converged = True
            break

        hist_idx = it % (history_len + 1)
        if hist_idx < history_len:
            hist_x[hist_idx] = x
            hist_y[hist_idx] = y
        else:
            window_dist = float(
                np.min(np.max(np.hypot(hist_x - x[None, :], hist_y - y[None, :]), axis=1))
            )
            if window_dist < config.convergence:
                converged = True
                break

    final_xy = np.column_stack([x, y])
    final_tangents, final_normals, final_curvatures = compute_contour_normals(
        final_xy, closed=closed
    )

    # Estimate average external energy
    avg_energy = float(np.mean(np.hypot(fx, fy)))

    return ActiveContourResult(
        xy=final_xy,
        tangents=final_tangents,
        normals=final_normals,
        curvatures=final_curvatures,
        converged=converged,
        iterations=iterations_run,
        energy=avg_energy,
        boundary_condition=boundary_condition,
    )


# -----------------------------------------------------------------------------
# Continuous Parametric B-Spline Snake (DropSnake Formulation)
# -----------------------------------------------------------------------------


def fit_bspline_snake(
    xy: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | tuple[float, float, float] | float | None = None,
    num_eval_points: int = 150,
    smoothing: float = 0.0,
    spline_degree: int = 3,
    closed: bool = False,
) -> BSplineSnakeResult:
    """Fit a continuous parametric cubic B-spline to drop contour points.

    Following Stalder et al. (2006, 2010), representing the droplet interface
    as an open B-spline curve provides C² continuity, exact sub-pixel derivatives,
    and closed-form contact angle calculation directly from endpoint tangents.

    Args:
        xy: (N, 2) array of contour coordinates (ordered along interface).
        substrate_line: Optional substrate line definition for baseline angle correction.
        num_eval_points: Number of points at which to evaluate the continuous spline.
        smoothing: Spline smoothing factor s (0 for interpolating spline).
        spline_degree: Spline polynomial degree k (default 3 for cubic).
        closed: Whether the curve is closed.

    Returns:
        BSplineSnakeResult containing evaluated points, analytical tangents,
        outward normals, curvatures, and contact angles in degrees.
    """
    xy_arr = np.asarray(xy, dtype=float)
    if len(xy_arr) < 4:
        raise ValueError(f"B-spline fitting requires at least 4 points, got {len(xy_arr)}")

    # Ensure endpoints are projected to substrate if provided
    if not closed and substrate_line is not None:
        xy_arr[0] = project_point_to_line(xy_arr[0], substrate_line)
        xy_arr[-1] = project_point_to_line(xy_arr[-1], substrate_line)

    k = min(spline_degree, len(xy_arr) - 1)
    tck, _ = splprep([xy_arr[:, 0], xy_arr[:, 1]], s=smoothing, k=k, per=closed)

    u = np.linspace(0.0, 1.0, num_eval_points)
    x_eval, y_eval = splev(u, tck, der=0)
    dx_eval, dy_eval = splev(u, tck, der=1)
    d2x_eval, d2y_eval = splev(u, tck, der=2)

    eval_xy = np.column_stack([x_eval, y_eval])

    # Analytical unit tangents
    tangent_lengths = np.hypot(dx_eval, dy_eval)
    tangent_lengths[tangent_lengths < 1e-9] = 1.0
    tx = dx_eval / tangent_lengths
    ty = dy_eval / tangent_lengths
    tangents = np.column_stack([tx, ty])

    # Outward unit normals: (ty, -tx)
    normals = np.column_stack([ty, -tx])

    # Analytical curvature: |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
    denom = (dx_eval**2 + dy_eval**2) ** 1.5
    denom[denom < 1e-9] = 1.0
    curvatures = (dx_eval * d2y_eval - dy_eval * d2x_eval) / denom

    # Contact angle calculation at endpoints
    contact_angles = None
    if not closed:
        u_sub = get_substrate_direction_vector(substrate_line, default_left_to_right=eval_xy)

        # Left endpoint (u = 0):
        # Interface tangent pointing into drop:
        t_left = np.array([dx_eval[0], dy_eval[0]], dtype=float)
        norm_l = float(np.hypot(t_left[0], t_left[1]))
        t_left /= (norm_l if norm_l > 1e-9 else 1.0)
        # Substrate vector pointing into drop footprint at left is +u_sub:
        cos_theta_l = float(np.dot(u_sub, t_left))
        theta_left = float(np.degrees(np.arccos(np.clip(cos_theta_l, -1.0, 1.0))))

        # Right endpoint (u = 1):
        # Interface tangent pointing into drop:
        t_right = np.array([-dx_eval[-1], -dy_eval[-1]], dtype=float)
        norm_r = float(np.hypot(t_right[0], t_right[1]))
        t_right /= (norm_r if norm_r > 1e-9 else 1.0)
        # Substrate vector pointing into drop footprint at right is -u_sub:
        cos_theta_r = float(np.dot(-u_sub, t_right))
        theta_right = float(np.degrees(np.arccos(np.clip(cos_theta_r, -1.0, 1.0))))

        contact_angles = (theta_left, theta_right)

    return BSplineSnakeResult(
        xy=eval_xy,
        tangents=tangents,
        normals=normals,
        curvatures=curvatures,
        contact_angles_deg=contact_angles,
        tck=tck,
    )
