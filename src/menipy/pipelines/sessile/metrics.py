"""Metrics.

Module implementation."""

from __future__ import annotations

import numpy as np

from menipy.common.geometry import (
    circle_fit_angle_at_point,
    cross2d,
    detect_baseline_ransac,
    find_contact_points_from_contour,
    refine_apex_curvature,
    tangent_angle_at_point,
    tangent_angle_at_point_pure,
)
from menipy.math.apex import detect_apex
from menipy.models.drop_extras import surface_area_mm2
from menipy.models.geometry import SubstrateProfile
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
    substrate_profile: SubstrateProfile | None = None,
) -> dict:
    """Compute sessile drop metrics with optional auto-detection and curved substrate support."""
    contour_2d = contour.reshape(-1, 2)

    # Reconcile substrate_line and substrate_profile
    baseline_confidence = 1.0
    if substrate_line is None and substrate_profile is None and auto_detect_baseline:
        p1_r, p2_r, baseline_confidence = detect_baseline_ransac(contour_2d)
        substrate_line = ((float(p1_r[0]), float(p1_r[1])), (float(p2_r[0]), float(p2_r[1])))
        substrate_profile = SubstrateProfile.from_line(
            substrate_line[0], substrate_line[1], confidence=baseline_confidence
        )
    elif substrate_profile is not None:
        baseline_confidence = float(substrate_profile.confidence)
        if substrate_line is None:
            substrate_line = substrate_profile.to_chord()
    elif substrate_line is not None:
        if isinstance(substrate_line, SubstrateProfile):
            substrate_profile = substrate_line
            substrate_line = substrate_profile.to_chord()
            baseline_confidence = float(substrate_profile.confidence)
        else:
            substrate_profile = SubstrateProfile.from_line(
                substrate_line[0], substrate_line[1], confidence=baseline_confidence
            )

    baseline_tilt_deg = 0.0
    if substrate_line is not None:
        line_p1 = np.asarray(substrate_line[0], dtype=float)
        line_p2 = np.asarray(substrate_line[1], dtype=float)
        line_vec = line_p2 - line_p1
        baseline_tilt_deg = float(np.degrees(np.arctan2(line_vec[1], line_vec[0])))

    contact_line = None
    diameter_px = 0.0
    height_px = 0.0
    volume_uL = 0.0
    contact_surface_mm2 = 0.0
    drop_surface_mm2 = 0.0

    # Use pre-computed contact points if provided, otherwise find them
    p1 = p2 = None
    sub_ref = substrate_profile if substrate_profile is not None else substrate_line
    if contact_points is not None:
        # Use pre-computed contact points from calibration
        p1 = np.array(contact_points[0], dtype=float)
        p2 = np.array(contact_points[1], dtype=float)
        contact_line = (tuple(p1.astype(int)), tuple(p2.astype(int)))
        diameter_px = float(np.linalg.norm(p1 - p2))
    elif sub_ref is not None:
        # Find contact points from contour intersection with substrate
        contour_2d = contour.reshape(-1, 2)
        p1, p2 = find_contact_points_from_contour(
            contour_2d, sub_ref, tolerance=contact_point_tolerance_px
        )
        if p1 is not None and p2 is not None:
            contact_line = (tuple(p1.astype(int)), tuple(p2.astype(int)))
            diameter_px = float(np.linalg.norm(p1 - p2))

    # Auto-detect apex if requested and not provided.
    apex_confidence = 1.0
    if apex is None and auto_detect_apex:
        apex_pt, apex_confidence = refine_apex_curvature(contour_2d)
        apex = (float(apex_pt[0]), float(apex_pt[1]))
        if contact_line is not None and p1 is not None and p2 is not None:
            x_min = float(min(p1[0], p2[0]))
            x_max = float(max(p1[0], p2[0]))
            pad = 0.1 * (x_max - x_min + 1.0)
            mask = (contour_2d[:, 0] >= (x_min - pad)) & (
                contour_2d[:, 0] <= (x_max + pad)
            )
            candidates = contour_2d[mask]
            if candidates.size > 0:
                fallback_res = detect_apex(
                    candidates,
                    mode="sessile",
                    baseline=contact_line,
                    substrate=substrate_profile,
                    refine=True,
                )
                fallback_apex = fallback_res.point
                if fallback_apex[1] < apex[1] - 1e-6:
                    apex = (float(fallback_apex[0]), float(fallback_apex[1]))
                    apex_confidence = min(apex_confidence, 0.6)

    # Calculate height and tilt-corrected diameter
    if substrate_profile is not None and substrate_profile.type != "line" and apex is not None:
        apex_pt = np.array(apex, dtype=float)
        y_sub_at_apex = substrate_profile.eval_y(float(apex_pt[0]))
        if y_sub_at_apex is not None:
            height_px = float(abs(y_sub_at_apex - apex_pt[1]))
        elif substrate_line is not None:
            p1_line = np.array(substrate_line[0])
            p2_line = np.array(substrate_line[1])
            num: float = float(np.abs(cross2d(p2_line - p1_line, p1_line - apex_pt)))
            den: float = float(np.linalg.norm(p2_line - p1_line))
            if den > 0:
                height_px = float(num / den)
    elif substrate_line is not None and apex is not None:
        p1_line = np.array(substrate_line[0])
        p2_line = np.array(substrate_line[1])
        apex_pt = np.array(apex)
        num: float = float(np.abs(cross2d(p2_line - p1_line, p1_line - apex_pt)))
        den: float = float(np.linalg.norm(p2_line - p1_line))
        if den > 0:
            height_px = float(num / den)

        line_vec = p2_line - p1_line
        line_len = np.linalg.norm(line_vec)
        if line_len > 0 and p1 is not None and p2 is not None:
            unit_line = line_vec / line_len
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
        side = np.sign(cross2d(v_sub, apex_pt - p1_sub))
        profile_mask = np.sign(cross2d(v_sub, contour_2d - p1_sub)) == side
        profile = contour_2d[profile_mask]

        if profile.size > 0 and len(profile) > 3:  # Need enough points for gradient
            # Project profile points onto the axis of symmetry to get coordinates for integration.
            vec_pa = profile - apex_pt
            # `z_coords` is the distance along the axis from the apex.
            z_coords_px = np.dot(vec_pa, v_axis)
            # `r_coords` is the perpendicular distance from the axis (the radius).
            r_coords_px = np.abs(cross2d(vec_pa, v_axis))
            contour_mm = np.column_stack([r_coords_px, z_coords_px]) / px_per_mm
            volume_uL = volume_from_contour(contour_mm)
            drop_surface_mm2 = surface_area_mm2(contour_mm * px_per_mm, px_per_mm)

    # Calculate contact angles using selected method
    theta_left_deg = 0.0
    theta_right_deg = 0.0
    uncertainty_left = 0.0
    uncertainty_right = 0.0
    method_left = method_right = "unavailable"
    selector_diagnostics: dict = {}
    lbadsa_payload: dict | None = None

    if sub_ref is not None and p1 is not None and p2 is not None:
        if contact_angle_method == "lbadsa":
            from menipy.common.lbadsa_solver import fit_lbadsa_drop

            lbadsa_payload = fit_lbadsa_drop(
                contour_2d,
                sub_ref,
                apex_xy=apex,
                contact_points=((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))),
                px_per_mm=px_per_mm,
                physics={"rho1": delta_rho + 1.2, "rho2": 1.2, "g": g},
                optimize_bo=True,
            )
            theta_left_deg = float(lbadsa_payload["theta_left_deg"])
            theta_right_deg = float(lbadsa_payload["theta_right_deg"])
            method_left = method_right = "lbadsa"
            uncertainty_left = uncertainty_right = float(lbadsa_payload.get("rmse_px", 1.0))
        elif contact_angle_method == "auto_residual":
            contour_len = len(contour.reshape(-1, 2))
            tangent_window_px = 30 if contour_len > 200 else 15
            tangent_weight_power = 2.0 if contour_len > 200 else 4.0

            def choose(point: np.ndarray) -> tuple[float, float, str, dict]:
                tangent_angle, tangent_rmse = tangent_angle_at_point_pure(
                    contour, point, sub_ref, window_px=tangent_window_px, weight_power=tangent_weight_power
                )
                circle_angle, circle_rmse = circle_fit_angle_at_point(contour, point, sub_ref)
                tangent_ok = bool(np.isfinite(tangent_angle) and np.isfinite(tangent_rmse) and tangent_rmse < 10.0)
                circle_ok = bool(np.isfinite(circle_angle) and np.isfinite(circle_rmse) and circle_rmse < 10.0)
                # Tangent remains the preferred/simple model unless circle is
                # materially better (20% lower residual).
                if circle_ok and (not tangent_ok or circle_rmse <= tangent_rmse * 0.8):
                    return circle_angle, circle_rmse, "circle_fit", {"tangent": {"angle_deg": tangent_angle, "rmse_px": tangent_rmse, "accepted": tangent_ok}, "circle_fit": {"angle_deg": circle_angle, "rmse_px": circle_rmse, "accepted": True}, "reason": "circle_materially_lower_residual"}
                if tangent_ok:
                    return tangent_angle, tangent_rmse, "tangent", {"tangent": {"angle_deg": tangent_angle, "rmse_px": tangent_rmse, "accepted": True}, "circle_fit": {"angle_deg": circle_angle, "rmse_px": circle_rmse, "accepted": circle_ok}, "reason": "tangent_preferred_or_tie"}
                return float("nan"), float("inf"), "rejected", {"tangent": {"angle_deg": tangent_angle, "rmse_px": tangent_rmse, "accepted": False}, "circle_fit": {"angle_deg": circle_angle, "rmse_px": circle_rmse, "accepted": circle_ok}, "reason": "no_valid_contact_angle_model"}

            theta_left_deg, uncertainty_left, method_left, diag_left = choose(p1)
            theta_right_deg, uncertainty_right, method_right, diag_right = choose(p2)
            selector_diagnostics = {"left": diag_left, "right": diag_right, "method_left": method_left, "method_right": method_right}
        elif contact_angle_method == "tangent":
            # Use tangent method
            contour_len = len(contour.reshape(-1, 2))
            tangent_window_px = 30 if contour_len > 200 else 15
            tangent_weight_power = 2.0 if contour_len > 200 else 4.0
            theta_left_deg, uncertainty_left = tangent_angle_at_point(
                contour,
                p1,
                sub_ref,
                window_px=tangent_window_px,
                weight_power=tangent_weight_power,
            )
            theta_right_deg, uncertainty_right = tangent_angle_at_point(
                contour,
                p2,
                sub_ref,
                window_px=tangent_window_px,
                weight_power=tangent_weight_power,
            )
        elif contact_angle_method == "circle_fit":
            # Use circle fit method
            theta_left_deg, uncertainty_left = circle_fit_angle_at_point(
                contour, p1, sub_ref
            )
            theta_right_deg, uncertainty_right = circle_fit_angle_at_point(
                contour, p2, sub_ref
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

    # Calculate local substrate slopes and apparent vs intrinsic angles
    alpha_left = 0.0
    alpha_right = 0.0
    if substrate_profile is not None:
        if p1 is not None:
            alpha_left = float(substrate_profile.eval_tangent_angle_deg(float(p1[0]), float(p1[1])))
        if p2 is not None:
            alpha_right = float(substrate_profile.eval_tangent_angle_deg(float(p2[0]), float(p2[1])))
    else:
        alpha_left = baseline_tilt_deg
        alpha_right = baseline_tilt_deg

    theta_left_apparent = float(theta_left_deg + alpha_left) if np.isfinite(theta_left_deg) else 0.0
    theta_right_apparent = float(theta_right_deg - alpha_right) if np.isfinite(theta_right_deg) else 0.0

    sub_type = substrate_profile.type if substrate_profile is not None else "line"
    sub_curv = float(substrate_profile.parameters.get("curvature_inv_px", 0.0)) * px_per_mm if (substrate_profile and px_per_mm > 0) else 0.0
    sub_radius = (float(substrate_profile.parameters.get("radius", 0.0)) / px_per_mm) if (substrate_profile and px_per_mm > 0 and sub_type == "circle_arc") else None
    sub_warn = bool(substrate_profile.confidence < 0.75) if substrate_profile is not None else (baseline_confidence < 0.75)
    sub_qual = "good" if (substrate_profile and substrate_profile.confidence >= 0.75) else ("fair" if (substrate_profile and substrate_profile.confidence >= 0.50) else "poor")

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
        "theta_left_apparent_deg": theta_left_apparent,
        "theta_right_apparent_deg": theta_right_apparent,
        "contact_surface_mm2": contact_surface_mm2,
        "drop_surface_mm2": drop_surface_mm2,
        "diameter_line": contact_line or ((0, 0), (0, 0)),
        "contact_line": contact_line,
        "substrate_profile": substrate_profile.model_dump() if substrate_profile is not None else None,
        "substrate_type": sub_type,
        "substrate_warning": sub_warn,
        "substrate_quality": sub_qual,
        "substrate_left_tilt_deg": alpha_left,
        "substrate_right_tilt_deg": alpha_right,
        "substrate_radius_mm": sub_radius,
        "substrate_curvature_inv_mm": sub_curv,
        "baseline_confidence": baseline_confidence,
        "baseline_tilt_deg": baseline_tilt_deg,
        "apex_confidence": apex_confidence,
        "baseline_method": baseline_method,
        "apex_method": apex_method,
        "method": contact_angle_method,
        **({"method_left": method_left, "method_right": method_right, "contact_angle_selector": selector_diagnostics} if contact_angle_method == "auto_residual" else {}),
        **({"experimental_geometry": {"sessile_contact_selector": {"accepted": method_left != "rejected" and method_right != "rejected", "rejection_reasons": (["left_contact_angle_no_valid_model"] if method_left == "rejected" else []) + (["right_contact_angle_no_valid_model"] if method_right == "rejected" else []), "method_left": method_left, "method_right": method_right}}} if contact_angle_method == "auto_residual" else {}),
        **({
            "bond_number": lbadsa_payload.get("bond_number"),
            "surface_tension_mN_m": lbadsa_payload.get("surface_tension_mN_m"),
            "R0_mm": lbadsa_payload.get("R0_mm"),
            "lbadsa_diagnostics": lbadsa_payload.get("diagnostics"),
            "lbadsa_model_contour_xy": lbadsa_payload.get("model_contour_xy"),
        } if lbadsa_payload is not None else {}),
        "uncertainty_deg": {"left": uncertainty_left, "right": uncertainty_right},
        "contact_angle_fit_rmse_px": {
            "left": uncertainty_left,
            "right": uncertainty_right,
        },
    }
