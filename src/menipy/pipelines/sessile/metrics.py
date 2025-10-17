from __future__ import annotations

import numpy as np


from menipy.common.geometry import find_contact_points_from_contour
from menipy.models.drop_extras import surface_area_mm2
from menipy.models.surface_tension import volume_from_contour

def compute_sessile_metrics(
    contour: np.ndarray,
    px_per_mm: float,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
    apex: tuple[int, int] | None = None,
    delta_rho: float = 998.8,
    g: float = 9.80665,
    contact_point_tolerance_px: float = 20.0,
) -> dict:
    """Placeholder for sessile metrics computation."""
    # This is a placeholder. The first real metrics to calculate are the
    # contact points and the base diameter.
    contact_line = None
    diameter_px = 0.0
    height_px = 0.0
    volume_uL = 0.0
    contact_surface_mm2 = 0.0
    drop_surface_mm2 = 0.0

    if substrate_line is not None:
        # The contour might be (N, 1, 2), so we reshape to (N, 2) for the helper.
        contour_2d = contour.reshape(-1, 2)
        p1, p2 = find_contact_points_from_contour(
            contour_2d, substrate_line, tolerance=contact_point_tolerance_px
        )

        if p1 is not None and p2 is not None:
            contact_line = (tuple(p1.astype(int)), tuple(p2.astype(int)))
            diameter_px = np.linalg.norm(p1 - p2)

        # Calculate height as the perpendicular distance from the apex to the substrate line.
        if apex is not None:
            p1_line = np.array(substrate_line[0])
            p2_line = np.array(substrate_line[1])
            apex_pt = np.array(apex)
            # Using the formula for the distance from a point to a line defined by two points.
            num = np.abs(np.cross(p2_line - p1_line, p1_line - apex_pt))
            den = np.linalg.norm(p2_line - p1_line)
            if den > 0:
                height_px = num / den

    diameter_mm = diameter_px / px_per_mm if px_per_mm > 0 else 0.0
    height_mm = height_px / px_per_mm if px_per_mm > 0 else 0.0
    contact_angle_deg = 0.0

    # Calculate contact surface area (base of the drop)
    if diameter_mm > 0:
        base_radius_mm = diameter_mm / 2.0
        contact_surface_mm2 = np.pi * (base_radius_mm**2)

    # Calculate volume by solid of revolution, correctly handling tilted substrates.
    if apex is not None and contact_line is not None and px_per_mm > 0:
        contour_2d = contour.reshape(-1, 2)

        # Define the axis of symmetry: a line through the apex, perpendicular to the substrate.
        p1_sub, p2_sub = np.array(contact_line[0]), np.array(contact_line[1])
        v_sub = p2_sub - p1_sub
        v_axis = np.array([-v_sub[1], v_sub[0]])  # Perpendicular vector
        v_axis = v_axis / (np.linalg.norm(v_axis) or 1)
        apex_pt = np.array(apex)

        # Filter for the droplet profile "above" the substrate line.
        # A point is "above" if the vector to it from the line has a positive dot product with the axis vector.
        side = np.sign(np.cross(v_sub, apex_pt - p1_sub))
        profile_mask = np.sign(np.cross(v_sub, contour_2d - p1_sub)) == side
        profile = contour_2d[profile_mask]

        if profile.size > 0:
            # Project profile points onto the axis of symmetry to get coordinates for integration.
            vec_pa = profile - apex_pt
            # `z_coords` is the distance along the axis from the apex.
            z_coords_px = np.dot(vec_pa, v_axis)
            # `r_coords` is the perpendicular distance from the axis (the radius).
            r_coords_px = np.abs(np.cross(vec_pa, v_axis))
            contour_mm = np.column_stack([r_coords_px, z_coords_px]) / px_per_mm
            volume_uL = volume_from_contour(contour_mm)
            drop_surface_mm2 = surface_area_mm2(contour_mm)

    # Calculate contact angle using the spherical cap approximation.
    # This is a simple geometric model. Other methods (tangential, polyfit,
    # Young-Laplace) can be added later.
    if diameter_mm > 0 and height_mm > 0:
        radius_mm = diameter_mm / 2.0
        # Formula for spherical cap: theta = 2 * arctan(h/r)
        theta_rad = 2 * np.arctan(height_mm / radius_mm)
        contact_angle_deg = np.degrees(theta_rad)

    return {
        "apex": apex or (0, 0),
        "diameter_mm": diameter_mm,
        "height_mm": height_mm,
        "volume_uL": volume_uL,
        "contact_angle_deg": contact_angle_deg,
        "contact_surface_mm2": contact_surface_mm2,
        "drop_surface_mm2": drop_surface_mm2,
        "diameter_line": contact_line or ((0, 0), (0, 0)),
        "contact_line": contact_line,
    }