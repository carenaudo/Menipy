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

    epsilon = 1e-9 * np.linalg.norm(line_vec) + 1e-9

    def is_inside(pt: np.ndarray) -> bool:
        return np.cross(line_vec, pt - p1) * side >= -epsilon

    intersections: list[np.ndarray] = []
    clipped: list[np.ndarray] = []

    prev = contour[-1]
    prev_in = is_inside(prev)

    # Fast path: everything already on the apex side.
    inside_mask = np.array([is_inside(pt) for pt in contour])
    if inside_mask.all():
        return contour, None

    for curr in contour:
        curr_in = is_inside(curr)

        if curr_in:
            if not prev_in:
                inter = _segment_intersection(prev, curr, p1, p2)
                if inter is not None:
                    intersections.append(inter)
                    clipped.append(inter)
            clipped.append(curr)
        elif prev_in:
            inter = _segment_intersection(prev, curr, p1, p2)
            if inter is not None:
                intersections.append(inter)
                clipped.append(inter)

        prev = curr
        prev_in = curr_in

    refined_contour = np.array(clipped)

    # Deduplicate intersections (may appear twice if segment touches line)
    unique_inters: list[np.ndarray] = []
    for pt in intersections:
        if not any(np.allclose(pt, u, atol=1e-9) for u in unique_inters):
            unique_inters.append(pt)

    contact_pts = None
    if len(unique_inters) >= 2:
        c1, c2 = unique_inters[:2]
        left_first = c1 if c1[0] <= c2[0] else c2
        right_second = c2 if c1[0] <= c2[0] else c1
        contact_pts = (tuple(map(float, left_first)), tuple(map(float, right_second)))

    return refined_contour, contact_pts


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
]
