from typing import Any

import numpy as np
import pytest

from menipy.common.geometry import cross2d
from menipy.pipelines.sessile.geometry import (
    _segment_intersection,
    clip_contour_to_substrate,
)


def legacy_clip(
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
    if (
        hasattr(substrate_line, "eval_y")
        and getattr(substrate_line, "type", "line") != "line"
    ):
        apex_pt = np.array(apex, dtype=float)
        apex_sub_y = substrate_line.eval_y(float(apex_pt[0]))
        if apex_sub_y is None:
            apex_sub_y = float(apex_pt[1]) + 10.0  # assume apex is above

        # In image coords, apex is typically above substrate (apex_y < apex_sub_y, sign = -1)
        drop_side = float(np.sign(apex_pt[1] - apex_sub_y))
        if abs(drop_side) < 1e-9:
            drop_side = -1.0  # default to drop above substrate

        sub_ys = np.array(
            [substrate_line.eval_y(float(pt[0])) for pt in contour], dtype=float
        )
        valid_mask = np.isfinite(sub_ys)
        if not np.any(valid_mask):
            return contour, None

        # Signed distance along vertical: negative means above substrate
        h_vals = contour[:, 1] - sub_ys
        epsilon = 0.5

        def is_point_inside(i: int) -> bool:
            if not valid_mask[i]:
                return True
            return bool(
                (h_vals[i] * drop_side >= -epsilon)
                if drop_side != 0
                else (h_vals[i] <= epsilon)
            )

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
                        inter = contour[curr_i] + t * (
                            contour[next_i] - contour[curr_i]
                        )
                        y_snap = substrate_line.eval_y(float(inter[0]))
                        if y_snap is not None:
                            inter[1] = y_snap
                        intersections.append(inter)
                        clipped.append(inter)

        refined_contour = (
            np.asarray(clipped, dtype=float).reshape(-1, 2)
            if clipped
            else np.empty((0, 2), dtype=float)
        )
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

    def is_inside(pt: np.ndarray) -> bool:
        """Check if inside."""
        # Signed area (cross product) test: positive means same side as apex
        # Keep a small negative tolerance to allow near-collinear points.
        return cross2d(line_vec, pt - p1) * side >= -epsilon

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


@pytest.mark.parametrize(
    "line", [((-2, 0), (2, 0)), ((-2, -1), (2, 1)), ((0, -2), (0, 2)), ((0, 0), (0, 0))]
)
@pytest.mark.parametrize(
    "case", ["random", "inside", "outside", "on_line", "small", "empty"]
)
def test_exact_clipping(line, case):
    rng = np.random.default_rng(42)
    points = rng.normal(size=(30, 2))
    if case == "inside":
        points[:, 1] -= 10
    elif case == "outside":
        points[:, 1] += 10
    elif case == "on_line":
        a, b = np.asarray(line)
        points = a + np.linspace(-1, 2, 30)[:, None] * (b - a)
        points[:, 1] += rng.choice([-1e-9, 0, 1e-9], 30)
    elif case == "small":
        points = points[:2]
    elif case == "empty":
        points = points[:0]
    original = points.copy()
    actual, contacts = clip_contour_to_substrate(points, line, (0, -5))
    expected, expected_contacts = legacy_clip(points, line, (0, -5))
    np.testing.assert_array_equal(actual, expected)
    assert contacts == expected_contacts
    np.testing.assert_array_equal(points, original)
