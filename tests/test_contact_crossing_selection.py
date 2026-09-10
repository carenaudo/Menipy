from typing import Any

import numpy as np
import pytest

from menipy.common.geometry import find_contact_points_from_contour


def legacy_contacts(
    contour: np.ndarray, contact_line: tuple | Any, tolerance: float = 20.0
) -> tuple:
    """Find left/right contact points on a droplet contour given a contact line or SubstrateProfile.

    Parameters
    ----------
    contour : np.ndarray
        Array shape (N,2) of contour points (x,y).
    contact_line : tuple or SubstrateProfile
        Pair of points (x1,y1), (x2,y2) describing the user-drawn contact line,
        or a SubstrateProfile object (line, circle_arc, polynomial).
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

    contour = np.asarray(contour, dtype=float)

    # Curved SubstrateProfile support (e.g. circle_arc, polynomial)
    if (
        hasattr(contact_line, "eval_y")
        and getattr(contact_line, "type", "line") != "line"
    ):
        sub_ys = np.array(
            [contact_line.eval_y(float(pt[0])) for pt in contour], dtype=float
        )
        valid_mask = np.isfinite(sub_ys)
        if not np.any(valid_mask):
            return (None, None)

        signed = contour[:, 1] - sub_ys
        intersections: list[np.ndarray] = []
        n = len(contour)
        for i in range(n):
            next_i = (i + 1) % n
            if not valid_mask[i] or not valid_mask[next_i]:
                continue
            h0 = signed[i]
            h1 = signed[next_i]
            dh = float(h1 - h0)
            if abs(dh) <= 1e-12:
                continue
            if h0 == 0.0 or (h0 * h1 < 0.0):
                t = float(-h0 / dh)
                if -1e-9 <= t <= 1.0 + 1e-9:
                    pt_inter = contour[i] + t * (contour[next_i] - contour[i])
                    snapped_y = contact_line.eval_y(float(pt_inter[0]))
                    if snapped_y is not None:
                        pt_inter[1] = snapped_y
                    intersections.append(pt_inter)

        if len(intersections) >= 2:
            intersections.sort(key=lambda p: float(p[0]))
            return (intersections[0], intersections[-1])

        abs_dist = np.abs(signed)
        abs_dist[~valid_mask] = np.inf
        candidate_idx = np.where(abs_dist <= tolerance)[0]
        if candidate_idx.size < 2:
            candidate_idx = np.argsort(abs_dist)[: max(2, int(0.02 * len(contour)))]
        if candidate_idx.size < 2:
            return (None, None)

        candidates = contour[candidate_idx]
        order = np.argsort(candidates[:, 0])
        left_pt = np.copy(candidates[order[0]])
        right_pt = np.copy(candidates[order[-1]])
        left_y = contact_line.eval_y(float(left_pt[0]))
        right_y = contact_line.eval_y(float(right_pt[0]))
        if left_y is not None:
            left_pt[1] = left_y
        if right_y is not None:
            right_pt[1] = right_y
        if np.linalg.norm(right_pt - left_pt) < 1e-6:
            return (None, None)
        return (left_pt, right_pt)

    if hasattr(contact_line, "to_chord"):
        contact_line = contact_line.to_chord()

    a = np.array(contact_line[0], dtype=float)
    b = np.array(contact_line[1], dtype=float)
    line_vec = b - a
    line_len = float(np.linalg.norm(line_vec))
    if line_len <= 1e-12:
        return (None, None)

    line_unit = line_vec / line_len
    normal = np.array([-line_unit[1], line_unit[0]], dtype=float)

    def project_to_line(pt: np.ndarray) -> np.ndarray:
        return a + line_unit * float(np.dot(pt - a, line_unit))

    def line_coord(pt: np.ndarray) -> float:
        return float(np.dot(pt - a, line_unit))

    signed = (contour - a) @ normal
    intersections: list[np.ndarray] = []
    for p0, p1, h0, h1 in zip(
        contour, np.roll(contour, -1, axis=0), signed, np.roll(signed, -1)
    ):
        dh = float(h1 - h0)
        if abs(dh) <= 1e-12:
            continue
        if h0 == 0.0 or h0 * h1 < 0.0:
            t = float(-h0 / dh)
            if -1e-9 <= t <= 1.0 + 1e-9:
                intersections.append(p0 + t * (p1 - p0))

    if len(intersections) >= 2:
        intersections.sort(key=line_coord)
        return (project_to_line(intersections[0]), project_to_line(intersections[-1]))

    dists = np.abs(signed)
    candidate_idx = np.where(dists <= tolerance)[0]
    if candidate_idx.size < 2:
        candidate_idx = np.argsort(dists)[: max(2, int(0.02 * len(contour)))]
    if candidate_idx.size < 2:
        return (None, None)

    candidates = contour[candidate_idx]
    coords = np.array([line_coord(pt) for pt in candidates], dtype=float)
    order = np.argsort(coords)
    left = project_to_line(candidates[order[0]])
    right = project_to_line(candidates[order[-1]])
    if np.linalg.norm(right - left) < 1e-6:
        return (None, None)
    return (left, right)


@pytest.mark.parametrize("count", [0, 1, 2, 10, 1000, 10000])
@pytest.mark.parametrize(
    "line",
    [
        ((-2, 0), (2, 0)),
        ((-2, -1), (2, 1)),
        ((0, -2), (0, 2)),
        ((0, 0), (0, 0)),
        ((-2, 20), (2, 20)),
    ],
)
def test_exact_contacts(count, line):
    rng = np.random.default_rng(42)
    contour = rng.normal(size=(count, 2))
    expected = legacy_contacts(contour, line)
    actual = find_contact_points_from_contour(contour, line)
    for a, b in zip(actual, expected):
        if b is None:
            assert a is None
        else:
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("height", [0.0, 1e-13, 1e-12, 2e-12])
def test_vertices_parallel_edges_and_closed_contours(height):
    contour = np.array(
        [[-2, height], [0, 0], [2, height], [2, 2], [-2, 2], [-2, height]]
    )
    for points in (contour, contour[::-1], np.repeat(contour, 2, axis=0)):
        expected = legacy_contacts(points, ((-3, 0), (3, 0)))
        actual = find_contact_points_from_contour(points, ((-3, 0), (3, 0)))
        for a, b in zip(actual, expected):
            if b is None:
                assert a is None
            else:
                np.testing.assert_array_equal(a, b)
