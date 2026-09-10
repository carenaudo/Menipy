"""Geometry.

Module implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from menipy.common.edge_detection import extract_external_contour
from menipy.common.geometry import cross2d
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
    r_x_s = float(cross2d(r, s))
    qp_x_s = float(cross2d(qp, s))

    if abs(r_x_s) < 1e-9:
        return None  # Parallel

    t = qp_x_s / r_x_s
    if 0.0 <= t <= 1.0:
        return p1 + t * r
    return None


def clip_contour_to_substrate(
    contour: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | Any,
    apex: tuple[float, float],
) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]] | None]:
    """
    Clip contour points that are below the substrate line (relative to apex).
    Find precise intersection points with the line or curved SubstrateProfile.

    Args:
        contour: (N, 2) array of points
        substrate_line: ((x1, y1), (x2, y2)) or SubstrateProfile
        apex: (x, y) - used to determine which side of the substrate is the drop

    Returns:
        refined_contour: (M, 2) array
        contact_points: ((x_left, y_left), (x_right, y_right)) or None
    """
    contour = np.asarray(contour, dtype=float)
    if len(contour) < 3:
        return contour, None

    # Curved SubstrateProfile clipping
    if hasattr(substrate_line, "eval_y") and getattr(substrate_line, "type", "line") != "line":
        apex_pt = np.array(apex, dtype=float)
        apex_sub_y = substrate_line.eval_y(float(apex_pt[0]))
        if apex_sub_y is None:
            apex_sub_y = float(apex_pt[1]) + 10.0  # assume apex is above

        # In image coords, apex is typically above substrate (apex_y < apex_sub_y, sign = -1)
        drop_side = float(np.sign(apex_pt[1] - apex_sub_y))
        if abs(drop_side) < 1e-9:
            drop_side = -1.0  # default to drop above substrate

        sub_ys = np.array([substrate_line.eval_y(float(pt[0])) for pt in contour], dtype=float)
        valid_mask = np.isfinite(sub_ys)
        if not np.any(valid_mask):
            return contour, None

        # Signed distance along vertical: negative means above substrate
        h_vals = contour[:, 1] - sub_ys
        epsilon = 0.5

        def is_point_inside(i: int) -> bool:
            if not valid_mask[i]:
                return True
            return bool((h_vals[i] * drop_side >= -epsilon) if drop_side != 0 else (h_vals[i] <= epsilon))

        inside_mask = np.array([is_point_inside(i) for i in range(len(contour))])
        if inside_mask.all():
            return contour, None

        intersections: list[np.ndarray] = []
        clipped: list[np.ndarray] = []
        n = len(contour)
        for i in range(n):
            curr_i = i
            next_i = (i + 1) % n
            prev_in = inside_mask[curr_i]
            curr_in = inside_mask[next_i]

            if prev_in:
                clipped.append(contour[curr_i])

            if prev_in != curr_in and valid_mask[curr_i] and valid_mask[next_i]:
                h0 = h_vals[curr_i]
                h1 = h_vals[next_i]
                dh = float(h1 - h0)
                if abs(dh) > 1e-12:
                    t = float(-h0 / dh)
                    if -1e-9 <= t <= 1.0 + 1e-9:
                        inter = contour[curr_i] + t * (contour[next_i] - contour[curr_i])
                        y_snap = substrate_line.eval_y(float(inter[0]))
                        if y_snap is not None:
                            inter[1] = y_snap
                        intersections.append(inter)
                        clipped.append(inter)

        refined_contour = np.asarray(clipped, dtype=float).reshape(-1, 2) if clipped else np.empty((0, 2), dtype=float)
        contact_pts = None
        if len(intersections) >= 2:
            intersections.sort(key=lambda p: float(p[0]))
            contact_pts = (
                (float(intersections[0][0]), float(intersections[0][1])),
                (float(intersections[-1][0]), float(intersections[-1][1])),
            )
        return refined_contour, contact_pts

    # Flat line handling
    if hasattr(substrate_line, "to_chord"):
        substrate_line = substrate_line.to_chord()

    p1 = np.array(substrate_line[0], dtype=float)
    p2 = np.array(substrate_line[1], dtype=float)
    apex_pt = np.array(apex, dtype=float)
    line_vec = p2 - p1

    # Keep the half-plane that contains the apex.
    side = float(np.sign(cross2d(line_vec, apex_pt - p1)))
    if abs(side) < 1e-9:
        # Apex exactly on the line; default to keeping the current contour.
        return contour, None

    # Tolerance scaled with substrate length (helps with pixel-scale geometry)
    epsilon = float(max(1e-9, 1e-6 * float(np.linalg.norm(line_vec))))

    intersections: list[np.ndarray] = []
    clipped: list[np.ndarray] = []

    # Fast path: everything already on the apex side.
    inside_mask = cross2d(line_vec, contour - p1) * side >= -epsilon
    if inside_mask.all():
        return contour, None

    # Iterate explicit segments (prev, curr) to avoid relying on implicit loop state
    rolled = np.roll(contour, -1, axis=0)
    for prev, curr, prev_in, curr_in in zip(
        contour, rolled, inside_mask, np.roll(inside_mask, -1)
    ):

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
    r_x_s = float(cross2d(r, s))
    if abs(r_x_s) < 1e-9:
        return None
    qp = q - p
    t = float(cross2d(qp, s) / r_x_s)
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
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | Any,
    apex: tuple[float, float],
    contact_points: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, tuple[tuple[float, float], tuple[float, float]] | None]:
    """Build calculation contour using side branches projected to substrate.

    Returns a synthetic contour that follows droplet side branches and closes with a
    single substrate segment or curved arc profile. Display contour should remain unchanged.
    """
    xy = np.asarray(contour, dtype=float).reshape(-1, 2)
    if len(xy) < 3:
        return xy, contact_points

    chord = substrate_line.to_chord() if hasattr(substrate_line, "to_chord") else substrate_line
    p1 = np.asarray(chord[0], dtype=float)
    p2 = np.asarray(chord[1], dtype=float)
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

    is_curved = hasattr(substrate_line, "eval_y") and getattr(substrate_line, "type", "line") != "line"

    def line_distance(pt: np.ndarray) -> float:
        return float(abs(cross2d(line_vec, pt - p1)) / line_len)

    if contact_points is not None and line_distance(cp_left_seed) <= 2.0:
        proj_left = cp_left_seed
    elif is_curved:
        y_left = substrate_line.eval_y(float(left_anchor[0]))
        proj_left = np.array([left_anchor[0], y_left if y_left is not None else left_anchor[1]], dtype=float)
    else:
        proj_left = _project_point_to_substrate_line(left_anchor, left_tangent, p1, p2)

    if contact_points is not None and line_distance(cp_right_seed) <= 2.0:
        proj_right = cp_right_seed
    elif is_curved:
        y_right = substrate_line.eval_y(float(right_anchor[0]))
        proj_right = np.array([right_anchor[0], y_right if y_right is not None else right_anchor[1]], dtype=float)
    else:
        proj_right = _project_point_to_substrate_line(right_anchor, right_tangent, p1, p2)

    # Keep deterministic left/right ordering on substrate coordinate.
    s_left = float(np.dot(proj_left - p1, line_dir))
    s_right = float(np.dot(proj_right - p1, line_dir))
    if s_left > s_right:
        proj_left, proj_right = proj_right, proj_left
        arc_xy = arc_xy[::-1]

    # Build synthetic contour: left contact -> side arc -> right contact -> substrate edge back to left.
    if is_curved:
        xs = np.linspace(proj_right[0], proj_left[0], 25)
        bottom_pts = []
        for x_val in xs:
            y_val = substrate_line.eval_y(float(x_val))
            if y_val is not None:
                bottom_pts.append([float(x_val), float(y_val)])
        bottom_sub = np.asarray(bottom_pts, dtype=float) if bottom_pts else proj_left.reshape(1, 2)
        calc_xy = np.vstack(
            [
                proj_left.reshape(1, 2),
                arc_xy,
                proj_right.reshape(1, 2),
                bottom_sub,
            ]
        )
    else:
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
