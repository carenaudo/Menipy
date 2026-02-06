"""
Contour smoothing utilities for droplet analysis.

This module provides Savitzky-Golay filtering for contour smoothing
and contact angle estimation from derivative-based tangent calculations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from menipy.models.context import Context

logger = logging.getLogger(__name__)


def filter_monotonic_contour(points: np.ndarray) -> np.ndarray:
    """Filter contour to ensure it is monotonic in X.
    
    If multiple Y values exist for the same X, keep the minimum Y (upper point).
    
    Args:
        points: (N, 2) array of contour points.
        
    Returns:
        Filtered (M, 2) array with unique X values.
    """
    if len(points) < 2:
        return points
        
    sorted_pts = points[np.argsort(points[:, 0])]
    unique_x, indices = np.unique(sorted_pts[:, 0], return_inverse=True)
    
    filtered_points = []
    for i in range(len(unique_x)):
        mask = (indices == i)
        y_values = sorted_pts[mask, 1]
        min_y = np.min(y_values)
        filtered_points.append([unique_x[i], min_y])
        
    return np.array(filtered_points)


def find_contact_intersections(
    x: np.ndarray,
    y: np.ndarray,
    substrate_y: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Find left and right contact points as curve-baseline intersections.
    
    Searches for zero-crossings of (y - substrate_y) from both ends
    of the curve and linearly interpolates the crossing point.
    
    Args:
        x: X coordinates (sorted).
        y: Y coordinates.
        substrate_y: Y value of the substrate line.
        
    Returns:
        Tuple of (left_contact, right_contact), each as (x, y) array or None.
    """
    d = y - substrate_y

    def locate_contact(x_arr, d_arr, from_left=True):
        """locate contact.

        Parameters
        ----------
        x_arr : type
        Description.
        d_arr : type
        Description.
        from_left : type
        Description.

        Returns
        -------
        type
        Description.
        """
        if not from_left:
            x_arr = x_arr[::-1]
            d_arr = d_arr[::-1]

        sign = np.sign(d_arr)
        # Find sign changes
        idx = np.where(sign[:-1] * sign[1:] < 0)[0]
        
        if len(idx) == 0:
            return None

        i = idx[0]  # First crossing
        
        # Linear interpolation
        denom = d_arr[i+1] - d_arr[i]
        if denom == 0:
            t = 0.5
        else:
            t = -d_arr[i] / denom
            
        x_c = x_arr[i] + t * (x_arr[i+1] - x_arr[i])
        return np.array([x_c, substrate_y])
    # locate_contact end

    left_cp = locate_contact(x, d, from_left=True)
    right_cp = locate_contact(x, d, from_left=False)

    return left_cp, right_cp


def smooth_contour(
    contour_xy: np.ndarray,
    substrate_y: float,
    window_length: int = 21,
    polyorder: int = 3,
    filter_monotonic: bool = False,
    filter_below_substrate: bool = True,
    extrapolate_contact_points: bool = True,
) -> Optional[Dict[str, Any]]:
    """Apply Savitzky-Golay smoothing to a droplet contour.
    
    Args:
        contour_xy: (N, 2) array of contour points.
        substrate_y: Y coordinate of the substrate line.
        window_length: Savgol window length (odd, >= polyorder + 2).
        polyorder: Polynomial order for the filter.
        filter_monotonic: If True, ensure one Y per X value.
        filter_below_substrate: If True, remove points below substrate.
        
    Returns:
        Dictionary with smoothing results:
            - x_smooth, y_smooth: Smoothed coordinates
        - apex: (x, y) tuple of the apex point
        - left_contact, right_contact: Contact points (or None)
        - left_slope, right_slope: Tangent slopes at contacts
        - left_angle_deg, right_angle_deg: Contact angles in degrees
        
        Returns None if smoothing fails (too few points, etc.)
    """
    dome_points = contour_xy.copy()
    
    # Filter points below substrate
    if filter_below_substrate:
        mask = dome_points[:, 1] <= substrate_y
        dome_points = dome_points[mask]
        
    if len(dome_points) < 5:
        logger.warning("Too few points for smoothing: %d", len(dome_points))
        return None
        
    if filter_monotonic:
        dome_points = filter_monotonic_contour(dome_points)
    
    # Ensure window is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Sort by x for monotonic curve
    order = np.argsort(dome_points[:, 0])
    x = dome_points[order, 0].astype(float)
    y = dome_points[order, 1].astype(float)
    
    # Handle cases where window is too large
    if len(x) < window_length:
        window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
    
    if window_length < polyorder + 2:
        logger.warning("Not enough points for smoothing: %d points, need %d", 
                      len(x), polyorder + 2)
        return None

    try:
        y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
        y_deriv = savgol_filter(y, window_length=window_length, polyorder=polyorder, 
                                deriv=1, delta=1.0)
    except ValueError as e:
        logger.warning("Savgol filter failed: %s", e)
        return None
    
    # Find apex (minimum Y, accounting for flat tops)
    min_y = np.min(y_smooth)
    min_idx = np.where(np.isclose(y_smooth, min_y))[0]
    if len(min_idx) > 1:
        apex_idx = int(min_idx[len(min_idx) // 2])
        apex_x = float(np.median(x[min_idx]))
        apex_y = float(min_y)
    else:
        apex_idx = int(min_idx[0])
        apex_x = float(x[apex_idx])
        apex_y = float(y_smooth[apex_idx])
    apex = (apex_x, apex_y)
    
    # Find contact points
    left_contact, right_contact = find_contact_intersections(x, y_smooth, substrate_y)
    
    
    # Calculate slopes at contact points
    left_slope = 0.0
    right_slope = 0.0
    
    if left_contact is not None:
        idx_l = np.argmin(np.abs(x - left_contact[0]))
        left_slope = float(y_deriv[idx_l])
    else:
        # Fallback: find closest point to substrate on left side
        left_subset_mask = x <= x[apex_idx]
        if np.any(left_subset_mask):
            lx = x[left_subset_mask]
            ly = y_smooth[left_subset_mask]
            ld = y_deriv[left_subset_mask]
            closest_i = np.argmin(np.abs(ly - substrate_y))
            left_contact = np.array([lx[closest_i], ly[closest_i]])
            left_slope = float(ld[closest_i])
        else:
            left_contact = np.array([x[0], y_smooth[0]])
            left_slope = float(y_deriv[0])

    if right_contact is not None:
        idx_r = np.argmin(np.abs(x - right_contact[0]))
        right_slope = float(y_deriv[idx_r])
    else:
        right_subset_mask = x >= x[apex_idx]
        if np.any(right_subset_mask):
            rx = x[right_subset_mask]
            ry = y_smooth[right_subset_mask]
            rd = y_deriv[right_subset_mask]
            closest_i = np.argmin(np.abs(ry - substrate_y))
            right_contact = np.array([rx[closest_i], ry[closest_i]])
            right_slope = float(rd[closest_i])
        else:
            right_contact = np.array([x[-1], y_smooth[-1]])
            right_slope = float(y_deriv[-1])
            
    # --- Extrapolation Logic ---
    if extrapolate_contact_points:
        # Left Extrapolation
        if left_contact is not None:
            dist_y = substrate_y - left_contact[1]
            is_floating = dist_y > 1.0 
            is_flat = abs(left_slope) < 0.2
            
            if is_floating or is_flat:
                target_y_start = left_contact[1] - 5
                target_y_end = left_contact[1] - 20
                mask = (x <= apex[0]) & (y_smooth < target_y_start) & (y_smooth > target_y_end)
                if np.sum(mask) > 3:
                    region_slopes = y_deriv[mask]
                    region_x = x[mask]
                    region_y = y_smooth[mask]
                    anchor_idx = np.argmax(region_y) 
                    anchor_x = region_x[anchor_idx]
                    anchor_y = region_y[anchor_idx]
                    anchor_slope = region_slopes[anchor_idx]
                    
                    if abs(anchor_slope) > 0.05:
                        new_x = anchor_x + (substrate_y - anchor_y) / anchor_slope
                        logger.debug(f"Extrapolating Left CP: Old=({left_contact[0]:.1f}, {left_contact[1]:.1f}) -> New=({new_x:.1f}, {substrate_y})")
                        left_contact = np.array([new_x, float(substrate_y)])
                        left_slope = float(anchor_slope)

        # Right Extrapolation
        if right_contact is not None:
            dist_y = substrate_y - right_contact[1]
            is_floating = dist_y > 1.0 
            is_flat = abs(right_slope) < 0.2
            if is_floating or is_flat:
                target_y_start = right_contact[1] - 5
                target_y_end = right_contact[1] - 20
                mask = (x >= apex[0]) & (y_smooth < target_y_start) & (y_smooth > target_y_end)
                if np.sum(mask) > 3:
                    region_slopes = y_deriv[mask]
                    region_x = x[mask]
                    region_y = y_smooth[mask]
                    anchor_idx = np.argmax(region_y) 
                    anchor_x = region_x[anchor_idx]
                    anchor_y = region_y[anchor_idx]
                    anchor_slope = region_slopes[anchor_idx]
                    if abs(anchor_slope) > 0.05:
                        new_x = anchor_x + (substrate_y - anchor_y) / anchor_slope
                        logger.debug(f"Extrapolating Right CP: Old=({right_contact[0]:.1f}, {right_contact[1]:.1f}) -> New=({new_x:.1f}, {substrate_y})")
                        right_contact = np.array([new_x, float(substrate_y)])
                        right_slope = float(anchor_slope)

    
    # Calculate contact angles from slopes
    # For left side: slope is typically negative (curve going up), angle = atan(|slope|)
    # For right side: slope is typically positive, angle = atan(|slope|)
    left_angle_deg = float(np.degrees(np.arctan(abs(left_slope))))
    right_angle_deg = float(np.degrees(np.arctan(abs(right_slope))))
    
    return {
        'x_smooth': x,
        'y_smooth': y_smooth,
        'apex': apex,
        'left_contact': left_contact,
        'right_contact': right_contact,
        'left_slope': left_slope,
        'right_slope': right_slope,
        'left_angle_deg': left_angle_deg,
        'right_angle_deg': right_angle_deg,
    }


def run(ctx: Context, settings=None) -> Context:
    """Apply contour smoothing as a pipeline stage.
    
    Reads contour from ctx.contour and substrate line from ctx.substrate_line,
    applies Savgol smoothing, and stores results in ctx.smoothing_results.
    
    Args:
        ctx: Pipeline context.
        settings: ContourSmoothingSettings or None (uses defaults).
        
    Returns:
        Updated context with smoothing_results dict.
    """
    # Import here to avoid circular imports
    from menipy.models.config import ContourSmoothingSettings
    
    if settings is None:
        settings = ContourSmoothingSettings()
    
    if not settings.enabled:
        return ctx
    
    # Get contour
    contour = getattr(ctx, 'contour', None)
    if contour is None or contour.xy is None:
        logger.warning("No contour available for smoothing")
        return ctx
        
    contour_xy = np.asarray(contour.xy)
    if contour_xy.size == 0:
        logger.warning("Empty contour, skipping smoothing")
        return ctx
    
    # Get substrate Y from substrate_line
    substrate_line = getattr(ctx, 'substrate_line', None)
    if substrate_line is None:
        # Fallback to baseline_y from geometry
        geometry = getattr(ctx, 'geometry', None)
        if geometry and getattr(geometry, 'baseline_y', None):
            substrate_y = geometry.baseline_y
        else:
            substrate_y = float(np.max(contour_xy[:, 1]))
            logger.info("No substrate line found, using max Y as baseline: %.1f", substrate_y)
    else:
        p1, p2 = substrate_line
        substrate_y = float((p1[1] + p2[1]) / 2)
    
    # Apply smoothing
    result = smooth_contour(
        contour_xy,
        substrate_y,
        window_length=settings.window_length,
        polyorder=settings.polyorder,
        filter_monotonic=settings.filter_monotonic,
        filter_below_substrate=settings.filter_below_substrate,
        extrapolate_contact_points=settings.extrapolate_contact_points,
    )
    
    if result is not None:
        ctx.smoothing_results = result
        logger.info("Contour smoothing applied: apex=(%.1f, %.1f), angles L=%.1f° R=%.1f°",
                   result['apex'][0], result['apex'][1],
                   result['left_angle_deg'], result['right_angle_deg'])
    else:
        ctx.smoothing_results = None
        logger.warning("Contour smoothing failed")
    
    return ctx
