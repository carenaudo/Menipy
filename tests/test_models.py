import numpy as np

from src.models.geometry import fit_circle
from src.models.properties import droplet_volume


def test_fit_circle():
    # Points on a circle centered at (1, 2) radius 3
    theta = np.linspace(0, 2 * np.pi, 50)
    x = 1 + 3 * np.cos(theta)
    y = 2 + 3 * np.sin(theta)
    center, radius = fit_circle(np.stack([x, y], axis=1))
    assert np.allclose(center, [1, 2], atol=1e-1)
    assert np.isclose(radius, 3, atol=1e-1)


def test_droplet_volume():
    volume = droplet_volume(2.0, np.deg2rad(60))
    assert volume > 0

