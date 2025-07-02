"""Property calculation functions."""

from __future__ import annotations

import numpy as np

from ..utils import pixels_to_mm, get_calibration


def droplet_volume(mask: np.ndarray) -> float:
    """Compute droplet volume from a binary mask using solid of revolution.

    Parameters
    ----------
    mask:
        2D array where non-zero pixels define the droplet silhouette.

    Returns
    -------
    float
        Volume in cubic millimetres based on the current calibration.
    """

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    cal = get_calibration()
    dy = 1.0 / cal.pixels_per_mm  # pixel height in mm
    volume = 0.0

    for row in mask:
        cols = np.where(row > 0)[0]
        if cols.size >= 2:
            width_px = cols[-1] - cols[0] + 1
            radius_mm = pixels_to_mm(float(width_px)) / 2.0
            volume += np.pi * radius_mm**2 * dy

    return float(volume)


