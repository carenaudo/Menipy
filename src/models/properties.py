"""Property calculation functions."""

from __future__ import annotations

import numpy as np

from ..utils import pixels_to_mm, get_calibration
from ..processing.segmentation import find_contours
from .geometry import fit_circle


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


def estimate_surface_tension(
    mask: np.ndarray, air_density: float, liquid_density: float
) -> float:
    """Estimate surface tension in mN/m from a binary droplet mask."""

    delta_rho = liquid_density - air_density
    if delta_rho <= 0:
        return 0.0

    contours = find_contours(mask)
    if not contours:
        return 0.0
    contour = max(contours, key=lambda c: c.shape[0]).astype(float)
    _, r0_px = fit_circle(contour)
    r0_mm = pixels_to_mm(float(r0_px))
    if r0_mm == 0:
        return 0.0

    ys, xs = np.nonzero(mask)
    if ys.size < 2 or xs.size < 2:
        return 0.0
    diameter_px = xs.max() - xs.min()
    r_max_mm = pixels_to_mm(float(diameter_px)) / 2.0

    gamma = delta_rho * 9.81 * (r_max_mm**2) / (2.0 * r0_mm)
    return float(gamma * 1000.0)


def contact_angle_from_mask(mask: np.ndarray) -> float:
    """Rough contact angle estimate from mask geometry."""

    ys, xs = np.nonzero(mask)
    if ys.size < 2 or xs.size < 2:
        return 0.0

    height_px = ys.max() - ys.min()
    radius_px = (xs.max() - xs.min()) / 2.0
    if radius_px == 0:
        return 0.0

    angle_rad = np.arctan2(height_px, radius_px)
    return float(np.degrees(angle_rad))


