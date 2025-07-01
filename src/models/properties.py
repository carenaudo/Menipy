"""Property calculation functions."""

import numpy as np


def droplet_volume(radius: float, contact_angle: float) -> float:
    """Approximate droplet volume using spherical cap formula."""
    h = radius * (1 - np.cos(contact_angle))
    volume = (np.pi * h**2 * (3 * radius - h)) / 3.0
    return volume

