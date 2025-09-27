import numpy as np

from menipy.common.geometry import fit_circle, horizontal_intersections
from menipy.models.properties import (
    droplet_volume,
    estimate_surface_tension,
    contact_angle_from_mask,
)


def test_fit_circle():
    # Points on a circle centered at (1, 2) radius 3
    theta = np.linspace(0, 2 * np.pi, 50)
    x = 1 + 3 * np.cos(theta)
    y = 2 + 3 * np.sin(theta)
    center, radius = fit_circle(np.stack([x, y], axis=1))
    assert np.allclose(center, [1, 2], atol=1e-1)
    assert np.isclose(radius, 3, atol=1e-1)


def test_droplet_volume():
    r = 10
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = ((x**2 + y**2) <= r**2).astype(np.uint8)
    volume = droplet_volume(mask, px_to_mm=1.0)
    expected = (4.0 / 3.0) * np.pi * r**3
    assert np.isclose(volume, expected, rtol=0.1)


def test_estimate_surface_tension_and_contact_angle():
    r = 10
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    mask = ((x ** 2 + y ** 2) <= r ** 2).astype(np.uint8)
    gamma = estimate_surface_tension(mask, 1.0, 1000.0, px_to_mm=1.0)
    angle = contact_angle_from_mask(mask)
    assert gamma > 0
    assert 0 < angle < 90


def test_horizontal_intersections():
    contour = np.array([[0, 1], [1, -1], [2, 1]], dtype=float)
    xs = horizontal_intersections(contour, 0.0)
    assert xs.size == 2
    assert np.allclose(xs, [0.5, 1.5])

