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
