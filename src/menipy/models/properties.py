"""Basic geometric and physical property calculations for droplets."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..zold_processing.segmentation import find_contours
from .geometry import fit_circle

GRAVITY = 9.81  # m s⁻²


def droplet_volume(mask: np.ndarray, px_to_mm: float, centred: bool = True) -> Optional[float]:
    """Return the droplet volume in mm³.

    The mask is treated as the silhouette of an axisymmetric drop. Each row is
    integrated as a disc whose radius is obtained from either the symmetry axis
    (if ``centred``) or from the row width.
    """

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if px_to_mm <= 0:
        raise ValueError("px_to_mm must be positive")
    if not np.any(mask):
        return None

    dy_mm = px_to_mm

    if centred:
        ys, xs = np.nonzero(mask)
        apex_y = ys.min()
        apex_cols = xs[ys == apex_y]
        x0 = int(np.median(apex_cols))
    else:
        x0 = None

    volume_mm3 = 0.0
    for y in range(mask.shape[0]):
        cols = np.where(mask[y] > 0)[0]
        if cols.size < 2:
            continue

        if centred and x0 is not None:
            radius_px = np.max(np.abs(cols - x0))
        else:
            radius_px = (cols[-1] - cols[0]) / 2.0

        r_mm = radius_px * px_to_mm
        volume_mm3 += np.pi * r_mm**2 * dy_mm

    return float(volume_mm3)


def estimate_surface_tension(
    mask: np.ndarray,
    air_density: float,
    liquid_density: float,
    px_to_mm: float,
    apex_window_px: int = 10,
) -> Optional[float]:
    """Return surface tension of the drop in mN m⁻¹.

    The apex radius is obtained from a circular fit over a thin strip above the
    highest point of the contour.
    """

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if px_to_mm <= 0:
        raise ValueError("px_to_mm must be positive")

    delta_rho = liquid_density - air_density
    if delta_rho <= 0:
        raise ValueError("liquid density must exceed air density")

    contours = find_contours(mask)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).astype(float)

    y_apex = contour[:, 1].min()
    apex_pts = contour[(contour[:, 1] - y_apex) < apex_window_px]
    if apex_pts.shape[0] < 3:
        return None

    (_, _), r0_px = fit_circle(apex_pts)

    r0_m = r0_px * px_to_mm * 1e-3
    if r0_m <= 0:
        return None

    ys, xs = np.nonzero(mask)
    if xs.size < 2:
        return None
    diameter_px = xs.max() - xs.min() + 1
    r_max_m = 0.5 * diameter_px * px_to_mm * 1e-3

    gamma_N_per_m = delta_rho * GRAVITY * r_max_m**2 / (2.0 * r0_m)
    return float(gamma_N_per_m * 1e3)


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


