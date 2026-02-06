"""Metrics.

Module implementation."""


from __future__ import annotations

import numpy as np

from menipy.common.geometry import fit_circle
from menipy.models.drop_extras import surface_area_mm2
from menipy.models.surface_tension import (
    jennings_pallas_beta,
    surface_tension,
    volume_from_contour,
)


def compute_pendant_metrics(
    contour: np.ndarray,
    px_per_mm: float,
    needle_diam_mm: float | None = None,
    apex: tuple[int, int] | None = None,
    delta_rho: float = 998.8,
    g: float = 9.80665,
    apex_window_px: int = 10,
) -> dict:
    """
    Compute metrics for a pendant drop including dimensions, surface tension, and volume.

    Calculates:
        - Height and Diameter (px and mm)
    - Shape factor (S) and Beta parameter
    - Surface Tension (using Jennings-Pallas method)
    - Volume and Surface Area (using solid of revolution)
    """
    height_px = 0.0
    diameter_px = 0.0
    diameter_line = ((0, 0), (0, 0))
    volume_uL = 0.0
    gamma_mN_m = 0.0
    beta = 0.0
    s1 = 0.0
    r0_px = 0.0
    needle_surface_mm2 = 0.0
    drop_surface_mm2 = 0.0

    if contour.size > 0:
        contour_2d = contour.reshape(-1, 2)
        y_coords = contour_2d[:, 1]
        x_coords = contour_2d[:, 0]

        # Height is the total vertical extent of the contour.
        height_px = y_coords.max() - y_coords.min()

        # Find the maximum diameter by checking the width at each unique y-level.
        unique_y = np.unique(y_coords)
        max_width = 0
        y_of_max_diam = 0
        for y_level in unique_y:
            points_at_level = x_coords[y_coords == y_level]
            if points_at_level.size > 1:
                current_width = points_at_level.max() - points_at_level.min()
                if current_width > max_width:
                    max_width = current_width
                    y_of_max_diam = y_level

        if max_width > 0:
            diameter_px = max_width
            points_at_max_diam = x_coords[y_coords == y_of_max_diam]
            x_min = points_at_max_diam.min()
            x_max = points_at_max_diam.max()
            diameter_line = (
                (int(x_min), int(y_of_max_diam)),
                (int(x_max), int(y_of_max_diam)),
            )

        # Calculate surface tension using the shape factor (s1) method.
        if apex is not None and px_per_mm > 0 and diameter_px > 0:
            # 1. Find radius of curvature at the apex (r0)
            apex_pts = contour_2d[(contour_2d[:, 1] - apex[1]) > -apex_window_px]
            if apex_pts.shape[0] >= 3:
                _, r0_px = fit_circle(apex_pts)

                # 2. Calculate shape factor s1 = De / (2 * r0)
                if r0_px > 0:
                    s1 = diameter_px / (2 * r0_px)
                    # 3. Calculate beta using the Jennings-Pallas correlation
                    beta = jennings_pallas_beta(s1)
                    # 4. Calculate surface tension
                    gamma_mN_m = surface_tension(delta_rho, g, r0_px / px_per_mm, beta)

        # Calculate volume and surface area by solid of revolution around the apex's vertical axis.
        if apex is not None and px_per_mm > 0:
            axis_x = apex[0]
            r_px = np.abs(contour_2d[:, 0] - axis_x)
            z_px = contour_2d[:, 1]
            contour_mm = np.column_stack([r_px, z_px]) / px_per_mm
            volume_uL = volume_from_contour(contour_mm)
            drop_surface_mm2 = surface_area_mm2(contour_mm * px_per_mm, px_per_mm)

    # Calculate needle surface area
    if needle_diam_mm is not None and needle_diam_mm > 0:
        needle_radius_mm = needle_diam_mm / 2.0
        needle_surface_mm2 = np.pi * (needle_radius_mm**2)

    height_mm = height_px / px_per_mm if px_per_mm > 0 else 0.0
    diameter_mm = diameter_px / px_per_mm if px_per_mm > 0 else 0.0

    return {
        "apex": apex or (0, 0),
        "diameter_mm": diameter_mm,
        "height_mm": height_mm,
        "volume_uL": volume_uL,
        "surface_tension_mN_m": gamma_mN_m,
        "beta": beta,
        "s1": s1,
        "r0_mm": r0_px / px_per_mm if px_per_mm > 0 else 0.0,
        "diameter_line": diameter_line,
        "needle_surface_mm2": needle_surface_mm2,
        "drop_surface_mm2": drop_surface_mm2,
    }
