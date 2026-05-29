"""Tests for test common geometry contact.

Unit tests."""

import numpy as np

from menipy.common.geometry import find_contact_points_from_contour


def test_find_contact_points_circle():
    # Create a circular contour centered at (50,50) with radius 20
    cx, cy, r = 50.0, 50.0, 20.0
    t = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
    contour = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])

    # User-drawn contact line near the bottom of the circle (horizontal)
    y_line = cy + r - 1.0
    contact_line = ((cx - 30, y_line), (cx + 30, y_line))

    left_pt, right_pt = find_contact_points_from_contour(
        contour, contact_line, tolerance=5.0
    )

    assert left_pt is not None and right_pt is not None
    # y should be close to the contact line y
    assert abs(left_pt[1] - y_line) < 3.0
    assert abs(right_pt[1] - y_line) < 3.0
    # points should lie on opposite sides of the center
    assert left_pt[0] < cx < right_pt[0]
    # distances from center should be approximately radius
    dleft = ((left_pt[0] - cx) ** 2 + (left_pt[1] - cy) ** 2) ** 0.5
    dright = ((right_pt[0] - cx) ** 2 + (right_pt[1] - cy) ** 2) ** 0.5
    assert abs(dleft - r) < 4.0
    assert abs(dright - r) < 4.0


def test_find_contact_points_ignores_closed_baseline_segment():
    theta = np.linspace(np.pi, 0.0, 120)
    dome = np.column_stack([100.0 + 50.0 * np.cos(theta), 100.0 - 35.0 * np.sin(theta)])
    baseline = np.column_stack([np.linspace(150.0, 50.0, 40), np.full(40, 100.0)])
    contour = np.vstack([dome, baseline])

    left_pt, right_pt = find_contact_points_from_contour(
        contour, ((40.0, 100.0), (160.0, 100.0)), tolerance=5.0
    )

    assert left_pt is not None and right_pt is not None
    np.testing.assert_allclose(left_pt, [50.0, 100.0], atol=1.0)
    np.testing.assert_allclose(right_pt, [150.0, 100.0], atol=1.0)


def test_find_contact_points_orders_tilted_substrate_along_line():
    substrate = ((0.0, 0.0), (120.0, 24.0))
    line_vec = np.array([120.0, 24.0])
    line_unit = line_vec / np.linalg.norm(line_vec)
    normal = np.array([-line_unit[1], line_unit[0]])
    coords = np.linspace(20.0, 100.0, 100)
    heights = 30.0 * np.sin(np.pi * (coords - 20.0) / 80.0)
    contour = np.array(
        [np.array(substrate[0]) + line_unit * s - normal * h for s, h in zip(coords, heights)]
    )

    left_pt, right_pt = find_contact_points_from_contour(
        contour, substrate, tolerance=4.0
    )

    assert left_pt is not None and right_pt is not None
    assert np.dot(left_pt - np.array(substrate[0]), line_unit) < np.dot(
        right_pt - np.array(substrate[0]), line_unit
    )
    assert abs(np.dot(left_pt - np.array(substrate[0]), normal)) < 1e-6
    assert abs(np.dot(right_pt - np.array(substrate[0]), normal)) < 1e-6
