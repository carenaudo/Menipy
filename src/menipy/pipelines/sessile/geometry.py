"""Geometry.

Module implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from menipy.common.edge_detection import extract_external_contour
from menipy.common.metrics import find_apex_index

from .metrics import compute_sessile_metrics


def _segment_intersection(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> np.ndarray | None:
    """Find intersection of line segment p1-p2 and line passing through q1-q2."""
    # Line p = p1 + t * (p2 - p1)
    # Line q = q1 + u * (q2 - q1)
    # Cross product (p - q1) x (q2 - q1) = 0 for intersection
    r = p2 - p1
    s = q2 - q1
    qp = q1 - p1
    r_x_s = float(np.cross(r, s))
    qp_x_s = float(np.cross(qp, s))

    if abs(r_x_s) < 1e-9:
        return None  # Parallel

    t = qp_x_s / r_x_s
    if 0.0 <= t <= 1.0:
        return p1 + t * r
    return None


def clip_contour_to_substrate(
    contour: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    apex: tuple[float, float],
) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]] | None]:
    """
    Clip contour points that are below the substrate line (relative to apex).
    Find precise intersection points with the line.

    Args:
        contour: (N, 2) array of points
        substrate_line: ((x1, y1), (x2, y2))
        apex: (x, y) - used to determine which side of the line is the drop

    Returns:
        refined_contour: (M, 2) array
        contact_points: ((x_left, y_left), (x_right, y_right)) or None
    """
    contour = np.asarray(contour, dtype=float)
    if len(contour) < 3:
        return contour, None

    p1 = np.array(substrate_line[0], dtype=float)
    p2 = np.array(substrate_line[1], dtype=float)
    apex_pt = np.array(apex, dtype=float)
    line_vec = p2 - p1

    # Keep the half-plane that contains the apex.
    side = float(np.sign(np.cross(line_vec, apex_pt - p1)))
    if abs(side) < 1e-9:
        # Apex exactly on the line; default to keeping the current contour.
        return contour, None

    1e-9 * float(np.linalg.norm(line_vec)) + 1e-9
    # Tolerance scaled with substrate length (helps with pixel-scale geometry)
    epsilon = float(max(1e-9, 1e-6 * float(np.linalg.norm(line_vec))))

    def is_inside(pt: np.ndarray) -> bool:
        """Check if inside.

        Parameters
        ----------
        pt : type
        Description.

        Returns
        -------
        type
        Description.
        """
        # Signed area (cross product) test: positive means same side as apex
        # Keep a small negative tolerance to allow near-collinear points.
        return np.cross(line_vec, pt - p1) * side >= -epsilon

    intersections: list[np.ndarray] = []
    clipped: list[np.ndarray] = []

    # Fast path: everything already on the apex side.
    inside_mask = np.array([is_inside(pt) for pt in contour])
    if inside_mask.all():
        return contour, None

    # Iterate explicit segments (prev, curr) to avoid relying on implicit loop state
    rolled = np.roll(contour, -1, axis=0)
    for prev, curr in zip(contour, rolled):
        prev_in = is_inside(prev)
        curr_in = is_inside(curr)

        if curr_in:
            if not prev_in:
                inter = _segment_intersection(prev, curr, p1, p2)
                if inter is not None:
                    intersections.append(inter)
                    clipped.append(inter)
            clipped.append(np.asarray(curr, dtype=float))
        elif prev_in:
            inter = _segment_intersection(prev, curr, p1, p2)
            if inter is not None:
                intersections.append(inter)
                clipped.append(inter)

    # Ensure refined_contour is always an (M, 2) float array (may be empty)
    if len(clipped) == 0:
        refined_contour: np.ndarray = np.empty((0, 2), dtype=float)
    else:
        refined_contour = np.asarray(clipped, dtype=float)
        if refined_contour.ndim == 1:
            # if a single 2-element point ended up as 1D, reshape
            if refined_contour.size == 2:
                refined_contour = refined_contour.reshape(1, 2)
            else:
                refined_contour = refined_contour.reshape(-1, 2)

    # Deduplicate intersections (may appear twice if segment touches line)
    unique_inters: list[np.ndarray] = []
    for pt in intersections:
        if not any(np.allclose(pt, u, atol=1e-9) for u in unique_inters):
            unique_inters.append(pt)

    contact_pts = None
    if len(unique_inters) >= 2:
        # Order intersections along substrate direction (robust for tilted/vertical lines)
        line_len = np.linalg.norm(line_vec)
        if line_len > 0:
            line_dir = line_vec / line_len
            unique_inters.sort(key=lambda p: float(np.dot(p - p1, line_dir)))
        else:
            unique_inters.sort(key=lambda p: float(p[0]))
        left_first = unique_inters[0]
        right_second = unique_inters[-1]
        contact_pts = (
            (float(left_first[0]), float(left_first[1])),
            (float(right_second[0]), float(right_second[1])),
        )

    return refined_contour, contact_pts


def _line_line_intersection(
    p: np.ndarray, r: np.ndarray, q: np.ndarray, s: np.ndarray
) -> np.ndarray | None:
    """Return intersection between two infinite lines p+t*r and q+u*s."""
    r_x_s = float(np.cross(r, s))
    if abs(r_x_s) < 1e-9:
        return None
    qp = q - p
    t = float(np.cross(qp, s) / r_x_s)
    return p + t * r


def _fit_local_tangent(contour: np.ndarray, idx: int, window: int = 5) -> np.ndarray | None:
    """Fit local tangent direction around contour index using PCA/SVD."""
    n = len(contour)
    if n < 3:
        return None
    if window < 1:
        window = 1

    idxs = [(idx + k) % n for k in range(-window, window + 1)]
    pts = contour[np.asarray(idxs, dtype=int)]
    if len(pts) < 3:
        return None

    centered = pts - np.mean(pts, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    direction = np.asarray(vh[0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None
    return direction / norm


def _project_point_to_substrate_line(
    point: np.ndarray,
    tangent_dir: np.ndarray | None,
    substrate_p1: np.ndarray,
    substrate_p2: np.ndarray,
) -> np.ndarray:
    """Project point onto substrate using tangent-line intersection with normal fallback."""
    sub_vec = substrate_p2 - substrate_p1
    sub_norm = float(np.linalg.norm(sub_vec))
    if sub_norm < 1e-9:
        return np.asarray(point, dtype=float)

    if tangent_dir is not None:
        inter = _line_line_intersection(
            np.asarray(point, dtype=float),
            np.asarray(tangent_dir, dtype=float),
            substrate_p1,
            sub_vec,
        )
        if inter is not None and np.isfinite(inter).all():
            return inter

    # Fallback: orthogonal projection to substrate line.
    unit_sub = sub_vec / sub_norm
    t = float(np.dot(point - substrate_p1, unit_sub))
    return substrate_p1 + t * unit_sub


def build_sessile_calculation_contour(
    contour: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    apex: tuple[float, float],
    contact_points: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]] | None]:
    """Build calculation contour using side branches projected to substrate.

    Returns a synthetic contour that follows droplet side branches and closes with a
    single substrate segment. Display contour should remain unchanged.
    """
    xy = np.asarray(contour, dtype=float).reshape(-1, 2)
    if len(xy) < 3:
        return xy, contact_points

    p1 = np.asarray(substrate_line[0], dtype=float)
    p2 = np.asarray(substrate_line[1], dtype=float)
    line_vec = p2 - p1
    line_len = float(np.linalg.norm(line_vec))
    if line_len < 1e-9:
        return xy, contact_points
    line_dir = line_vec / line_len

    apex_pt = np.asarray(apex, dtype=float)
    apex_idx = int(np.argmin((xy[:, 0] - apex_pt[0]) ** 2 + (xy[:, 1] - apex_pt[1]) ** 2))

    # Use provided contacts as side anchors when available, otherwise infer by x-extrema.
    if contact_points is not None:
        cp_left_seed = np.asarray(contact_points[0], dtype=float)
        cp_right_seed = np.asarray(contact_points[1], dtype=float)
    else:
        left_i = int(np.argmin(xy[:, 0]))
        right_i = int(np.argmax(xy[:, 0]))
        cp_left_seed = xy[left_i]
        cp_right_seed = xy[right_i]

    left_idx = int(np.argmin(np.sum((xy - cp_left_seed) ** 2, axis=1)))
    right_idx = int(np.argmin(np.sum((xy - cp_right_seed) ** 2, axis=1)))

    n = len(xy)

    def path_indices(start: int, end: int) -> np.ndarray:
        if start <= end:
            return np.arange(start, end + 1, dtype=int)
        return np.concatenate(
            [np.arange(start, n, dtype=int), np.arange(0, end + 1, dtype=int)]
        )

    p1_idx = path_indices(left_idx, right_idx)
    p2_idx = path_indices(right_idx, left_idx)
    arc_idx = p1_idx if apex_idx in set(p1_idx.tolist()) else p2_idx
    arc_xy = xy[arc_idx]

    if len(arc_xy) < 2:
        return xy, contact_points

    # Ensure arc orientation is left -> right along substrate direction.
    s0 = float(np.dot(arc_xy[0] - p1, line_dir))
    s1 = float(np.dot(arc_xy[-1] - p1, line_dir))
    if s0 > s1:
        arc_xy = arc_xy[::-1]

    left_anchor = np.asarray(arc_xy[0], dtype=float)
    right_anchor = np.asarray(arc_xy[-1], dtype=float)
    left_anchor_idx = int(np.argmin(np.sum((xy - left_anchor) ** 2, axis=1)))
    right_anchor_idx = int(np.argmin(np.sum((xy - right_anchor) ** 2, axis=1)))

    left_tangent = _fit_local_tangent(xy, left_anchor_idx, window=5)
    right_tangent = _fit_local_tangent(xy, right_anchor_idx, window=5)

    def line_distance(pt: np.ndarray) -> float:
        return float(abs(np.cross(line_vec, pt - p1)) / line_len)

    if contact_points is not None and line_distance(cp_left_seed) <= 2.0:
        proj_left = cp_left_seed
    else:
        proj_left = _project_point_to_substrate_line(left_anchor, left_tangent, p1, p2)

    if contact_points is not None and line_distance(cp_right_seed) <= 2.0:
        proj_right = cp_right_seed
    else:
        proj_right = _project_point_to_substrate_line(right_anchor, right_tangent, p1, p2)

    # Keep deterministic left/right ordering on substrate coordinate.
    s_left = float(np.dot(proj_left - p1, line_dir))
    s_right = float(np.dot(proj_right - p1, line_dir))
    if s_left > s_right:
        proj_left, proj_right = proj_right, proj_left
        arc_xy = arc_xy[::-1]

    # Build synthetic contour: left contact -> side arc -> right contact -> substrate edge back to left.
    calc_xy = np.vstack(
        [
            proj_left.reshape(1, 2),
            arc_xy,
            proj_right.reshape(1, 2),
            proj_left.reshape(1, 2),
        ]
    )

    projected_contacts = (
        (float(proj_left[0]), float(proj_left[1])),
        (float(proj_right[0]), float(proj_right[1])),
    )
    return calc_xy, projected_contacts


@dataclass
class HelperBundle:
    px_per_mm: float
    substrate_line: tuple[tuple[int, int], tuple[int, int]] | None = None
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None = None
    delta_rho: float = 998.8
    g: float = 9.80665
    contact_point_tolerance_px: float = 20.0


@dataclass
class SessileMetrics:
    contour: np.ndarray
    apex: tuple[int, int]
    diameter_line: tuple[tuple[int, int], tuple[int, int]]
    contact_line: tuple[tuple[int, int], tuple[int, int]] | None
    diameter_center: tuple[int, int] | None
    derived: dict[str, float]


def analyze(frame: np.ndarray, helpers: HelperBundle) -> SessileMetrics:
    """Return sessile-drop metrics and geometry from ``frame``."""
    contour = extract_external_contour(frame)
    apex_idx = find_apex_index(contour, "sessile")
    apex = tuple(contour[apex_idx].astype(int))
    metrics = compute_sessile_metrics(
        contour.astype(float),
        px_per_mm=helpers.px_per_mm,
        substrate_line=helpers.substrate_line,
        apex=apex,
        contact_point_tolerance_px=helpers.contact_point_tolerance_px,
    )
    return SessileMetrics(
        contour=contour,
        apex=apex,
        diameter_line=metrics["diameter_line"],
        contact_line=metrics.get("contact_line"),
        diameter_center=metrics.get("diameter_center"),
        derived=metrics,
    )


class SessilePipeline:
    """A pipeline for analyzing sessile drops."""

    name = "sessile"


__all__ = [
    "analyze",
    "SessileMetrics",
    "HelperBundle",
    "SessilePipeline",
    "clip_contour_to_substrate",
    "build_sessile_calculation_contour",
]
