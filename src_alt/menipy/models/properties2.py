"""Property calculation functions."""

from __future__ import annotations
from typing import Optional
import numpy as np
import cv2
from ..utils import pixels_to_mm, get_calibration
from ..processing.segmentation import find_contours
from .geometry import fit_circle
GRAVITY = 9.81  # m s⁻²

def droplet_volume(
    mask: np.ndarray,
    px_to_mm: float = pixels_to_mm(float(1)),
    centred: bool = True,
) -> Optional[float]:
    """
    Compute droplet volume (mm³) from a binary mask using the
    disc-integration (solid-of-revolution) method. 
    Same as before, fixed using ChatGPT o3 and just paste the original code to fix the theoretical errors.

    This method assumes the droplet is axisymmetric and approximates it
    as a series of discs (horizontal slices) with varying radii.

    Parameters
    ----------
    mask : 2-D ndarray[bool or uint8]
        Non-zero pixels define the droplet silhouette.
    px_to_mm : float
        Conversion factor: millimetres per pixel.
    centred : bool, default True
        If True, radii are measured from the horizontal symmetry axis
        (the column of the apex).  If False, radius = half of row width.

    Returns
    -------
    float or None
        Volume in cubic millimetres, or None if it cannot be determined.
    """
    # -------- Basic checks ---------------------------------------------------
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if not np.any(mask):
        return None

    # Height of one pixel slice in millimetres
    dy_mm = px_to_mm

    # -------- Optional: find symmetry axis (column) --------------------------
    if centred:
        ys, xs = np.nonzero(mask)
        # apex is smallest y; use median x of those top-row pixels
        apex_y = ys.min()
        apex_cols = xs[ys == apex_y]
        x0 = int(np.median(apex_cols))
    else:
        x0 = None  # not used

    volume_mm3 = 0.0
    for y in range(mask.shape[0]):
        row = mask[y]
        cols = np.where(row)[0]
        if cols.size < 2:
            continue  # skip rows with 0 or 1 foreground pixel

        if centred and x0 is not None:
            # radius = max horizontal distance from symmetry axis
            radius_px = cols - x0
            r_px = radius_px[np.argmax(np.abs(radius_px))]
            r_px = abs(r_px)
        else:
            # radius = half of total width
            r_px = (cols[-1] - cols[0]) / 2.0  # drop +1 for sub-pixel interp

        r_mm = r_px * px_to_mm
        volume_mm3 += np.pi * r_mm**2 * dy_mm

    return float(volume_mm3)

# def droplet_volume(mask: np.ndarray) -> float:
#     """Compute droplet volume from a binary mask using solid of revolution.

#     Parameters
#     ----------
#     mask:
#         2D array where non-zero pixels define the droplet silhouette.

#     Returns
#     -------
#     float
#         Volume in cubic millimetres based on the current calibration.
#     """

#     if mask.ndim != 2:
#         raise ValueError("mask must be a 2D array")

#     cal = get_calibration()
#     dy = 1.0 / cal.pixels_per_mm  # pixel height in mm
#     volume = 0.0

#     for row in mask:
#         cols = np.where(row > 0)[0]
#         if cols.size >= 2:
#             width_px = cols[-1] - cols[0] + 1
#             radius_mm = pixels_to_mm(float(width_px)) / 2.0
#             volume += np.pi * radius_mm**2 * dy

#     return float(volume)


def estimate_surface_tension(
    mask: np.ndarray,
    air_density: float,
    liquid_density: float,
    px_to_mm: float= pixels_to_mm(float(1)),
    apex_window_px: int = 10,
) -> Optional[float]:
    """
    Estimate surface tension (mN m⁻¹) from a binary droplet mask.
    Made with the assumption that the droplet is axisymmetric and
    approximated by a circle at the apex.
    Fixed using ChatGPT o3 and just paste the original code to fix the theoretical errors.

    Parameters
    ----------
    mask : 2-D bool/uint8
        Binary image where the droplet pixels are True/1.
    air_density, liquid_density : float
        Densities in kg m⁻³.
    px_to_mm : float
        Calibration factor: millimetres per pixel.
    apex_window_px : int, optional
        Height of the strip (in pixels) above the apex used for circle fitting.

    Returns
    -------
    float or None
        Surface tension in mN m⁻¹, or None if it cannot be estimated.
    """
    Δρ = liquid_density - air_density
    print(f"Δρ = {Δρ} kg/m³, air: {air_density} kg/m³, liquid: {liquid_density} kg/m³")
    if Δρ <= 0:
        raise ValueError("liquid density must exceed air density")

    # --- Obtain the outer contour and ensure it is the correct one -----------
    contours = find_contours(mask)  # your own helper
    if not contours:
        return None
    # pick by *area* to discard holes
    contour = max(contours, key=cv2.contourArea).astype(float)

    # --- Apex circle fit -----------------------------------------------------
    # keep only a thin horizontal strip around the apex (smallest y)
    y_apex = contour[:, 1].min()
    apex_pts = contour[(contour[:, 1] - y_apex) < apex_window_px]

    if apex_pts.shape[0] < 3:
        return None
    (_, _), r0_px = fit_circle(apex_pts)  # returns (center, radius)

    r0_mm = r0_px * px_to_mm
    if r0_mm <= 0:
        return None

    # --- Max equatorial radius ----------------------------------------------
    ys, xs = np.nonzero(mask)
    if xs.size < 2:
        return None
    d_px = (xs.max() - xs.min() + 1)  # add 1 for inclusive width
    r_max_mm = 0.5 * d_px * px_to_mm
    print(f"r0_mm = {r0_mm} mm, r_max_mm = {r_max_mm} mm")

    # --- Young–Laplace one-parameter approximation --------------------------
    gamma_N_per_m = Δρ * GRAVITY * (r_max_mm * 1e-3) ** 2 / (2 * r0_mm * 1e-3)
    gamma_mN_per_m = gamma_N_per_m * 1e3
    return float(gamma_mN_per_m)

# def estimate_surface_tension(
#     mask: np.ndarray, air_density: float, liquid_density: float
# ) -> float:
#     """Estimate surface tension in mN/m from a binary droplet mask."""

#     delta_rho = liquid_density - air_density
#     if delta_rho <= 0:
#         return 0.0

#     contours = find_contours(mask)
#     if not contours:
#         return 0.0
#     contour = max(contours, key=lambda c: c.shape[0]).astype(float)
#     _, r0_px = fit_circle(contour)
#     r0_mm = pixels_to_mm(float(r0_px))
#     if r0_mm == 0:
#         return 0.0

#     ys, xs = np.nonzero(mask)
#     if ys.size < 2 or xs.size < 2:
#         return 0.0
#     diameter_px = xs.max() - xs.min()
#     r_max_mm = pixels_to_mm(float(diameter_px)) / 2.0

#     gamma = delta_rho * 9.81 * (r_max_mm**2) / (2.0 * r0_mm)
#     return float(gamma * 1000.0)


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


