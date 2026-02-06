"""Metrics.

Module implementation."""


from __future__ import annotations

import numpy as np
from typing import cast


from menipy.common.geometry import (
    find_contact_points_from_contour,
    detect_baseline_ransac,
    refine_apex_curvature,
    tangent_angle_at_point,
    circle_fit_angle_at_point,
)
from menipy.models.drop_extras import surface_area_mm2
from menipy.models.surface_tension import volume_from_contour


def compute_sessile_metrics(
    contour: np.ndarray,
    px_per_mm: float,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
    apex: tuple[float, float] | None = None,
    delta_rho: float = 998.8,
    g: float = 9.80665,
    contact_point_tolerance_px: float = 20.0,
    auto_detect_baseline: bool = False,
    auto_detect_apex: bool = False,
    contact_angle_method: str = "tangent",
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> dict:
    """Compute sessile drop metrics with optional auto-detection."""
    contour_2d = contour.reshape(-1, 2)

    # Auto-detect baseline if requested and not provided
    baseline_confidence = 1.0
    if substrate_line is None and auto_detect_baseline:
        p1, p2, baseline_confidence = detect_baseline_ransac(contour_2d)
        substrate_line = ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))

    contact_line = None
    diameter_px = 0.0
    height_px = 0.0
    volume_uL = 0.0
    contact_surface_mm2 = 0.0
    drop_surface_mm2 = 0.0

    # Use pre-computed contact points if provided, otherwise find them
    p1 = p2 = None
    if contact_points is not None:
        # Use pre-computed contact points from calibration
        p1 = np.array(contact_points[0], dtype=float)
        p2 = np.array(contact_points[1], dtype=float)
        contact_line = (tuple(p1.astype(int)), tuple(p2.astype(int)))
        diameter_px = float(np.linalg.norm(p1 - p2))
    elif substrate_line is not None:
        # Find contact points from contour intersection with substrate
        contour_2d = contour.reshape(-1, 2)
        p1, p2 = find_contact_points_from_contour(
            contour_2d, substrate_line, tolerance=contact_point_tolerance_px
        )
        if p1 is not None and p2 is not None:
            contact_line = (tuple(p1.astype(int)), tuple(p2.astype(int)))
            diameter_px = float(np.linalg.norm(p1 - p2))

    # Auto-detect apex if requested and not provided. If we have contact points,
    # prefer an apex near the midpoint between contacts (min y in that region).
    apex_confidence = 1.0
    if apex is None and auto_detect_apex:
        apex_pt, apex_confidence = refine_apex_curvature(contour_2d)
        apex = (float(apex_pt[0]), float(apex_pt[1]))
        # Post-process apex: if contact_line available, ensure apex is centered
        # above the substrate between the contact points; if not, pick the
        # minimum-y point between contacts as a robust fallback.
        if contact_line is not None and p1 is not None and p2 is not None:
            x_min = float(min(p1[0], p2[0]))
            x_max = float(max(p1[0], p2[0]))
            # Slightly expand search window to tolerate detection noise
            pad = 0.1 * (x_max - x_min + 1.0)
            mask = (contour_2d[:, 0] >= (x_min - pad)) & (contour_2d[:, 0] <= (x_max + pad))
            candidates = contour_2d[mask]
            if candidates.size > 0:
                min_idx = int(np.argmin(candidates[:, 1]))
                fallback_apex = candidates[min_idx]
                # Accept fallback if it is vertically above the refined apex
                if fallback_apex[1] < apex[1] - 1e-6:
                    apex = (float(fallback_apex[0]), float(fallback_apex[1]))
                    apex_confidence = min(apex_confidence, 0.6)

    # Calculate height and tilt-corrected diameter whenever we have an apex and
    # a substrate line (should not be limited only to cases where apex was
    # auto-detected).
    if substrate_line is not None and apex is not None:
        p1_line = np.array(substrate_line[0])
        p2_line = np.array(substrate_line[1])
        apex_pt = np.array(apex)
        # Using the formula for the distance from a point to a line defined by two points.
        num: float = float(np.abs(np.cross(p2_line - p1_line, p1_line - apex_pt)))
        den: float = float(np.linalg.norm(p2_line - p1_line))
        if den > 0:
            height_px = float(num / den)

        # Use tilt-corrected coordinates for diameter if substrate is tilted
        line_vec = p2_line - p1_line
        line_len = np.linalg.norm(line_vec)
        if line_len > 0 and p1 is not None and p2 is not None:
            # Unit vector along the line
            unit_line = line_vec / line_len
            # Perpendicular unit vector
            unit_perp = np.array([-unit_line[1], unit_line[0]])

            # Project contact points onto tilt-corrected coordinate system
            p1_proj = float(np.dot(p1 - p1_line, unit_line))
            p2_proj = float(np.dot(p2 - p1_line, unit_line))
            diameter_px = float(abs(p2_proj - p1_proj))

    diameter_mm = diameter_px / px_per_mm if px_per_mm > 0 else 0.0
    height_mm = height_px / px_per_mm if px_per_mm > 0 else 0.0
    contact_angle_deg = 0.0  # Initialize for legacy compatibility

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

        if profile.size > 0 and len(profile) > 3:  # Need enough points for gradient
            # Project profile points onto the axis of symmetry to get coordinates for integration.
            vec_pa = profile - apex_pt
            # `z_coords` is the distance along the axis from the apex.
            z_coords_px = np.dot(vec_pa, v_axis)
            # `r_coords` is the perpendicular distance from the axis (the radius).
            r_coords_px = np.abs(np.cross(vec_pa, v_axis))
            contour_mm = np.column_stack([r_coords_px, z_coords_px]) / px_per_mm
            volume_uL = volume_from_contour(contour_mm)
            drop_surface_mm2 = surface_area_mm2(contour_mm * px_per_mm, px_per_mm)

    # Calculate contact angles using selected method
    theta_left_deg = 0.0
    theta_right_deg = 0.0
    uncertainty_left = 0.0
    uncertainty_right = 0.0

    if substrate_line is not None and p1 is not None and p2 is not None:
        if contact_angle_method == "tangent":
            # Use tangent method
            theta_left_deg, uncertainty_left = tangent_angle_at_point(
                contour, p1, substrate_line
            )
            theta_right_deg, uncertainty_right = tangent_angle_at_point(
                contour, p2, substrate_line
            )
        elif contact_angle_method == "circle_fit":
            # Use circle fit method
            theta_left_deg, uncertainty_left = circle_fit_angle_at_point(
                contour, p1, substrate_line
            )
            theta_right_deg, uncertainty_right = circle_fit_angle_at_point(
                contour, p2, substrate_line
            )
        elif contact_angle_method == "spherical_cap":
            # Use spherical cap approximation (legacy)
            if diameter_mm > 0 and height_mm > 0:
                radius_mm = diameter_mm / 2.0
                theta_rad = 2 * np.arctan(height_mm / radius_mm)
                contact_angle_deg = np.degrees(theta_rad)
                theta_left_deg = contact_angle_deg
                theta_right_deg = contact_angle_deg
                # Estimate uncertainty based on geometric approximation
                uncertainty_left = uncertainty_right = 2.0  # Rough estimate
        else:
            # Default to spherical cap
            if diameter_mm > 0 and height_mm > 0:
                radius_mm = diameter_mm / 2.0
                theta_rad = 2 * np.arctan(height_mm / radius_mm)
                contact_angle_deg = np.degrees(theta_rad)
                theta_left_deg = contact_angle_deg
                theta_right_deg = contact_angle_deg
                uncertainty_left = uncertainty_right = 2.0
    else:
        # Fallback to spherical cap if no substrate/contact points
        if diameter_mm > 0 and height_mm > 0:
            radius_mm = diameter_mm / 2.0
            theta_rad = 2 * np.arctan(height_mm / radius_mm)
            contact_angle_deg = np.degrees(theta_rad)
            theta_left_deg = contact_angle_deg
            theta_right_deg = contact_angle_deg
            uncertainty_left = uncertainty_right = 2.0

    # Determine method tags
    baseline_method = (
        "auto_ransac"
        if auto_detect_baseline and substrate_line is not None
        else "manual"
    )
    apex_method = (
        "auto_curvature" if auto_detect_apex and apex is not None else "manual"
    )

    return {
        "apex": apex or (0, 0),
        "diameter_mm": diameter_mm,
        "height_mm": height_mm,
        "volume_uL": volume_uL,
        "contact_angle_deg": (
            (theta_left_deg + theta_right_deg) / 2
            if theta_left_deg > 0 and theta_right_deg > 0
            else contact_angle_deg
        ),  # Legacy compatibility
        "theta_left_deg": theta_left_deg,
        "theta_right_deg": theta_right_deg,
        "contact_surface_mm2": contact_surface_mm2,
        "drop_surface_mm2": drop_surface_mm2,
        "diameter_line": contact_line or ((0, 0), (0, 0)),
        "contact_line": contact_line,
        "baseline_confidence": baseline_confidence,
        "apex_confidence": apex_confidence,
        "baseline_method": baseline_method,
        "apex_method": apex_method,
        "method": contact_angle_method,
        "uncertainty_deg": {"left": uncertainty_left, "right": uncertainty_right},
    }
