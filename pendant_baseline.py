#!/usr/bin/env python3
"""
Pendant Drop Autocalibration and Strict Young-Laplace Fitting Baseline Script.
This script is self-contained and does not import from the Menipy package structure.
It directly copies/adapts the exact physical and mathematical models used in Menipy.
"""

from __future__ import annotations

import os
import sys
import math
import logging
from dataclasses import dataclass
from typing import Any, Tuple, List, Optional, Dict
from pathlib import Path

import cv2
import numpy as np
from scipy.integrate import trapezoid, solve_ivp
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pendant_baseline")

# Physical constants matching Menipy defaults
RHO1 = 1000.0        # drop density (water) in kg/m3
RHO2 = 1.2           # fluid/phase density (air) in kg/m3
G = 9.80665          # gravity in m/s2
NEEDLE_DIAM_MM = 1.83 # physical needle diameter in mm

# ==============================================================================
# 1. GEOMETRIC & PHYSICAL HELPER FUNCTIONS (From Menipy common/models)
# ==============================================================================

def fit_circle(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a circle to 2D points using linear least squares."""
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    center = c[:2]
    radius = np.sqrt(c[2] + center.dot(center))
    return center, radius


def jennings_pallas_beta(s1: float) -> float:
    """Return the dimensionless form factor beta."""
    a3, a2, a1, a0 = 0.41727, -1.0908, 1.3906, 0.005306
    return ((a3 * s1 + a2) * s1 + a1) * s1 + a0


def surface_tension(delta_rho: float, g: float, r0_mm: float, beta: float) -> float:
    """Return surface tension in Newton per metre."""
    r0 = r0_mm / 1000.0
    return delta_rho * g * r0**2 / beta


def bond_number(delta_rho: float, g: float, r0_mm: float, gamma: float) -> float:
    """Return dimensionless Bond number."""
    r0 = r0_mm / 1000.0
    return delta_rho * g * r0**2 / gamma


def vmax_uL(gamma_N_m: float, needle_diam_mm: float, delta_rho: float, g: float = 9.80665) -> float:
    """Max detachment volume (µL) from Berry et al. (2015)."""
    D = needle_diam_mm / 1000.0
    vmax_m3 = math.pi * gamma_N_m * D / (delta_rho * g)
    return vmax_m3 * 1e9


def worthington_number(vol_uL: float, vmax_uL: float) -> float:
    """Worthington number Wo = V / Vmax."""
    return vol_uL / vmax_uL


# ==============================================================================
# 2. AUTOCALIBRATION & DETECTIONS (From Menipy common/auto_calibrator.py)
# ==============================================================================

def run_pendant_autocalibration(image: np.ndarray) -> dict:
    """
    Perform automatic segmentation and region/needle/apex/contour detection
    for a pendant drop image, matching AutoCalibrator._detect_pendant.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    height, width = gray.shape[:2]
    image_area = height * width
    
    # Apply CLAHE for contrast enhancement (for preview/alignment checking)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # 1. Segment using Otsu thresholding (better for high-contrast silhouettes)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 2. Find the main drop contour (largest, centered)
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find any contours in the image.")
        
    img_center_x = width // 2
    min_area = image_area * 0.05  # At least 5% of image area
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            # Must be roughly centered (within 30% of center)
            if abs(cx - img_center_x) < (width * 0.3):
                valid_contours.append((cnt, area))
                
    if not valid_contours:
        raise ValueError("No valid pendant drop contour found (must be centered and >= 5% of image area).")
        
    # Select largest valid contour
    drop_cnt = max(valid_contours, key=lambda x: x[1])[0]
    
    # 3. Detect needle and contact points
    x, y, w, h = cv2.boundingRect(drop_cnt)
    pts = drop_cnt.reshape(-1, 2)
    
    # Define needle shaft reference (top 20 pixels)
    top_limit = y + 20
    
    # Left shaft line: median X of points in top-left quadrant
    left_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] < (x + w / 2))]
    if len(left_shaft_pts) == 0:
        raise ValueError("Failed to locate left needle shaft line.")
    ref_x_left = np.median(left_shaft_pts[:, 0])
    
    # Right shaft line: median X of points in top-right quadrant
    right_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] > (x + w / 2))]
    if len(right_shaft_pts) == 0:
        raise ValueError("Failed to locate right needle shaft line.")
    ref_x_right = np.median(right_shaft_pts[:, 0])
    
    # Trace down column by column to find where the contour deviates from the shaft (contact points)
    tolerance = 3
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [drop_cnt], -1, 255, 1)
    
    # Find left contact point (where contour moves left of shaft)
    contact_y_left = y
    contact_x_left = int(ref_x_left)
    for cy in range(y, y + h):
        row = mask[cy, 0 : int(x + w / 2)]
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            current_x = indices[0]
            if current_x < (ref_x_left - tolerance):
                contact_y_left = cy
                contact_x_left = current_x
                break
                
    # Find right contact point (where contour moves right of shaft)
    contact_y_right = y
    contact_x_right = int(ref_x_right)
    for cy in range(y, y + h):
        row = mask[cy, int(x + w / 2) : width]
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            current_x = indices[-1] + int(x + w / 2)
            if current_x > (ref_x_right + tolerance):
                contact_y_right = cy
                contact_x_right = current_x
                break
                
    # Needle bottom is the higher of the two contact points
    needle_bottom = min(contact_y_left, contact_y_right)
    needle_x = int(ref_x_left)
    needle_y = y
    needle_w = int(ref_x_right - ref_x_left)
    needle_h = needle_bottom - y
    
    # 4. Find apex (bottom of pendant drop, maximum Y)
    apex_idx = np.argmax(pts[:, 1])
    apex_pt = pts[apex_idx]
    apex_point = (int(apex_pt[0]), int(apex_pt[1]))
    
    # 5. Compute ROI from detected regions
    x_min = int(np.min(pts[:, 0]))
    x_max = int(np.max(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    y_max = max(int(np.max(pts[:, 1])), apex_point[1])
    
    pad = 20
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min)
    x_max = min(width, x_max + pad)
    y_max = min(height, y_max + pad)
    roi_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    return {
        "drop_contour": pts.astype(np.float64),
        "needle_rect": (needle_x, needle_y, needle_w, needle_h),
        "contact_points": (
            (contact_x_left, contact_y_left),
            (contact_x_right, contact_y_right)
        ),
        "apex_point": apex_point,
        "roi_rect": roi_rect,
        "binary_mask": binary_clean,
        "enhanced_image": enhanced_gray
    }

# ==============================================================================
# 3. CONTOUR EXTRAC & PROCESSING (From Menipy pipelines/pendant/stages.py)
# ==============================================================================

def _contour_to_xy(contour: object) -> np.ndarray:
    xy = np.asarray(contour, dtype=float)
    if xy.ndim == 3 and xy.shape[-1] == 2:
        xy = xy.reshape(-1, 2)
    elif xy.ndim == 2 and xy.shape[1] >= 2:
        xy = xy[:, :2]
    else:
        xy = xy.reshape(-1, 2)
    return np.asarray(xy, dtype=float)


def _clip_contour_at_pendant_contacts(xy: np.ndarray, contact_points: object | None) -> np.ndarray:
    if contact_points is None:
        return xy
    try:
        contacts = np.asarray(contact_points, dtype=float).reshape(-1, 2)
    except Exception:
        return xy
        
    if contacts.shape[0] < 2 or xy.size == 0:
        return xy
        
    contact_y = float(np.min(contacts[:2, 1]))
    clipped = xy[xy[:, 1] >= contact_y]
    if clipped.shape[0] < 3:
        return xy
    return np.vstack([contacts[:2], clipped])


def _pendant_max_width(xy: np.ndarray, contact_points: object | None = None) -> tuple[float, tuple[tuple[int, int], tuple[int, int]]]:
    if xy.size == 0:
        return 0.0, ((0, 0), (0, 0))
    measured = xy
    if contact_points is not None:
        try:
            contacts = np.asarray(contact_points, dtype=float).reshape(-1, 2)
            if contacts.shape[0] >= 2:
                contact_y = float(np.min(contacts[:2, 1]))
                candidate = xy[xy[:, 1] >= contact_y]
                if candidate.shape[0] >= 2:
                    measured = candidate
        except Exception:
            measured = xy

    row_keys = np.rint(measured[:, 1]).astype(int)
    best_width = 0.0
    best_line = ((0, 0), (0, 0))
    for row in np.unique(row_keys):
        xs = measured[row_keys == row, 0]
        if xs.size < 2:
            continue
        width = float(np.max(xs) - np.min(xs))
        if width > best_width:
            x_min = float(np.min(xs))
            x_max = float(np.max(xs))
            best_width = width
            best_line = ((int(round(x_min)), int(row)), (int(round(x_max)), int(row)))

    if best_width <= 0.0 and measured.shape[0] >= 2:
        x_min = float(np.min(measured[:, 0]))
        x_max = float(np.max(measured[:, 0]))
        y_mid = int(round(float(np.median(measured[:, 1]))))
        best_width = x_max - x_min
        best_line = ((int(round(x_min)), y_mid), (int(round(x_max)), y_mid))

    return best_width, best_line


def _pendant_apex_radius_px(xy: np.ndarray, apex_xy: tuple[float, float] | None, window_px: float = 20.0) -> float:
    if apex_xy is None or xy.size == 0:
        return 0.0
    apex_y = float(apex_xy[1])
    apex_pts = xy[(xy[:, 1] - apex_y) > -float(window_px)]
    if apex_pts.shape[0] < 3:
        return 0.0
    _, radius = fit_circle(apex_pts)
    if not np.isfinite(radius) or radius <= 0:
        return 0.0
    return float(radius)


def _radial_profile_integrals(profile_mm: np.ndarray) -> tuple[float, float]:
    """Return (volume_uL, surface_mm2) for a radial (r, z) profile in millimetres."""
    profile = np.asarray(profile_mm, dtype=float).reshape(-1, 2)
    if profile.shape[0] < 3:
        return 0.0, 0.0
    profile = profile[np.all(np.isfinite(profile), axis=1)]
    profile = profile[profile[:, 0] >= 0]
    if profile.shape[0] < 3:
        return 0.0, 0.0
        
    order = np.argsort(profile[:, 1])
    z_mm = profile[order, 1]
    r_mm = profile[order, 0]
    z_mm, unique_idx = np.unique(z_mm, return_index=True)
    r_mm = r_mm[unique_idx]
    if z_mm.shape[0] < 3 or float(np.ptp(z_mm)) <= 0:
        return 0.0, 0.0
        
    volume_uL = float(np.pi * trapezoid(r_mm**2, z_mm))
    dr_dz = np.gradient(r_mm, z_mm, edge_order=2)
    surface_mm2 = float(2.0 * np.pi * trapezoid(r_mm * np.sqrt(1.0 + dr_dz**2), z_mm))
    return abs(volume_uL), abs(surface_mm2)

# ==============================================================================
# 4. YOUNG-LAPLACE SOLVER & optimizer (From pipelines/pendant/strict_young_laplace.py)
# ==============================================================================

def build_pendant_profile_envelope_mm(
    contour_px: np.ndarray,
    *,
    axis_x_px: float,
    apex_y_px: float,
    px_per_mm: float,
    bin_px: float = 1.0,
) -> np.ndarray:
    """Collapse a pendant contour into a radial (r_mm, z_mm) envelope starting at apex (0,0)."""
    xy = np.asarray(contour_px, dtype=float).reshape(-1, 2)
    if xy.shape[0] < 3 or px_per_mm <= 0:
        return np.empty((0, 2), dtype=float)

    bin_px = max(float(bin_px), 1.0)
    row_keys = np.rint(xy[:, 1] / bin_px).astype(int)
    rows: list[tuple[float, float]] = []
    for row in np.unique(row_keys):
        pts = xy[row_keys == row]
        if pts.size == 0:
            continue
        y_px = float(np.mean(pts[:, 1]))
        z_mm = (float(apex_y_px) - y_px) / float(px_per_mm)
        if z_mm < -0.5 / float(px_per_mm):
            continue
        xs = pts[:, 0]
        if xs.size >= 2:
            r_mm = (float(np.max(xs)) - float(np.min(xs))) / (2.0 * px_per_mm)
        else:
            r_mm = float(np.max(np.abs(xs - float(axis_x_px)))) / float(px_per_mm)
        if np.isfinite(z_mm) and np.isfinite(r_mm) and r_mm >= 0:
            rows.append((r_mm, max(0.0, z_mm)))

    if not rows:
        return np.empty((0, 2), dtype=float)

    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 1])]
    merged: list[tuple[float, float]] = []
    for z in np.unique(arr[:, 1]):
        r = float(np.max(arr[arr[:, 1] == z, 0]))
        merged.append((r, float(z)))
    profile = np.asarray(merged, dtype=float)
    if profile.shape[0] < 2:
        return np.empty((0, 2), dtype=float)

    if profile[0, 1] > 1e-9:
        profile = np.vstack([[0.0, 0.0], profile])
    else:
        profile[0, 1] = 0.0
        profile[0, 0] = min(
            profile[0, 0], profile[1, 0] if profile.shape[0] > 1 else 0.0
        )
    return profile


def pendant_contour_to_model_mm(
    contour_px: np.ndarray,
    *,
    axis_x_px: float,
    apex_y_px: float,
    px_per_mm: float,
) -> np.ndarray:
    """Convert image-space pendant contour points into apex-centered mm."""
    xy = np.asarray(contour_px, dtype=float).reshape(-1, 2)
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    x_mm = (xy[:, 0] - float(axis_x_px)) / float(px_per_mm)
    z_mm = (float(apex_y_px) - xy[:, 1]) / float(px_per_mm)
    return np.column_stack([x_mm, z_mm])


def model_mm_to_pendant_px(
    model_mm: np.ndarray,
    *,
    axis_x_px: float,
    apex_y_px: float,
    px_per_mm: float,
) -> np.ndarray:
    """Convert apex-centered strict model coordinates back to image pixels."""
    xy = np.asarray(model_mm, dtype=float).reshape(-1, 2)
    x_px = xy[:, 0] * float(px_per_mm) + float(axis_x_px)
    y_px = float(apex_y_px) - xy[:, 1] * float(px_per_mm)
    return np.column_stack([x_px, y_px])


def integrate_young_laplace_profile_mm(
    r0_mm: float,
    beta: float,
    *,
    target_height_mm: float | None = None,
    needle_radius_mm: float | None = None,
    max_step: float = 0.02,
    branch: str = "full",
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Integrate a symmetric pendant Young-Laplace profile in millimetres."""
    r0_mm = float(r0_mm)
    beta = float(beta)
    if not np.isfinite(r0_mm) or not np.isfinite(beta) or r0_mm <= 0:
        profile = np.empty((0, 2), dtype=float)
        meta = {"stop_reason": "invalid_parameters"}
        return (profile, meta) if return_metadata else profile

    z_target = None
    if target_height_mm is not None and target_height_mm > 0:
        z_target = max(float(target_height_mm) / r0_mm, 0.2)

    r_needle_target = None
    if needle_radius_mm is not None and needle_radius_mm > 0:
        r_needle_target = float(needle_radius_mm) / r0_mm

    def ode(_s: float, y: np.ndarray) -> list[float]:
        r, z, psi = y
        if abs(r) < 1e-10:
            sin_psi_over_r = 1.0
        else:
            sin_psi_over_r = float(np.sin(psi) / r)
        return [
            float(np.cos(psi)),
            float(np.sin(psi)),
            float(2.0 - beta * z - sin_psi_over_r),
        ]

    def hit_axis(s: float, y: np.ndarray) -> float:
        if s <= 0.1:
            return 1.0
        return float(y[0] - 1e-6)

    hit_axis.terminal = True
    hit_axis.direction = -1

    events = [hit_axis]
    if z_target is not None:
        def hit_target_height(_s: float, y: np.ndarray) -> float:
            return float(y[1] - z_target)
        hit_target_height.terminal = True
        hit_target_height.direction = 1
        events.append(hit_target_height)

    if r_needle_target is not None:
        def hit_needle_radius_after_equator(_s: float, y: np.ndarray) -> float:
            r, _z, psi = y
            if psi <= (np.pi / 2.0):
                return 1.0
            return float(r - r_needle_target)
        hit_needle_radius_after_equator.terminal = True
        hit_needle_radius_after_equator.direction = -1
        events.append(hit_needle_radius_after_equator)

    s_max = max(8.0, (z_target or 4.0) * 3.0 + 2.0)
    sol = solve_ivp(
        ode,
        (0.0, s_max),
        [0.0, 0.0, 0.0],
        method="RK45",
        events=events,
        max_step=max_step,
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success or sol.y.shape[1] < 3:
        profile = np.empty((0, 2), dtype=float)
        meta = {"stop_reason": "solver_failed", "solver_message": str(sol.message)}
        return (profile, meta) if return_metadata else profile

    stop_reason = "s_max"
    if sol.t_events:
        event_names = ["axis_return"]
        if z_target is not None:
            event_names.append("height_cutoff")
        if r_needle_target is not None:
            event_names.append("needle_radius")
        for name, events_for_name in zip(event_names, sol.t_events):
            if len(events_for_name) > 0:
                stop_reason = name
                break

    r_right = sol.y[0] * r0_mm
    z_right = sol.y[1] * r0_mm
    if branch == "right":
        profile = np.column_stack([r_right, z_right])
        meta = {"stop_reason": stop_reason, "solver_message": str(sol.message)}
        return (profile, meta) if return_metadata else profile

    r_left = -r_right[::-1]
    z_left = z_right[::-1]
    r_full = np.concatenate([r_left[:-1], r_right])
    z_full = np.concatenate([z_left[:-1], z_right])
    profile = np.column_stack([r_full, z_full])
    meta = {"stop_reason": stop_reason, "solver_message": str(sol.message)}
    return (profile, meta) if return_metadata else profile


def _normal_projection_residuals_mm(obs_mm: np.ndarray, model_mm: np.ndarray) -> np.ndarray:
    """Project observed points to nearest model normals and return mm distances."""
    if model_mm.shape[0] < 3:
        return np.full(obs_mm.shape[0], 1e3, dtype=float)

    tangents = np.gradient(model_mm, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    _, idx = cKDTree(model_mm).query(obs_mm)
    return np.einsum("ij,ij->i", obs_mm - model_mm[idx], normals[idx]).astype(float)


def _bounds_from_seed(
    *,
    r0_seed_mm: float,
    beta_seed: float,
    diameter_mm: float,
    height_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r0_seed_mm = max(float(r0_seed_mm), 0.05)
    beta_seed = float(np.clip(beta_seed, 0.02, 4.0))
    r0_upper = max(20.0, r0_seed_mm * 5.0, diameter_mm * 3.0, height_mm * 3.0)
    offset_x = max(0.25, diameter_mm * 0.15)
    offset_z = max(0.25, height_mm * 0.15)
    lower = np.array([max(0.02, r0_seed_mm / 5.0), 1e-3, -offset_x, -offset_z])
    upper = np.array([r0_upper, 5.0, offset_x, offset_z])
    x0 = np.array([r0_seed_mm, beta_seed, 0.0, 0.0])
    return x0, lower, upper


def _is_pinned(params: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> bool:
    span = np.maximum(upper - lower, 1e-12)
    tol = np.maximum(1e-5, span * 1e-3)
    return bool(np.any(params <= lower + tol) or np.any(params >= upper - tol))


def fit_pendant_young_laplace_strict(
    contour_px: np.ndarray,
    axis_x_px: float,
    apex_y_px: float,
    px_per_mm: float,
    r0_seed_mm: float,
    beta_seed: float,
    physics: dict[str, Any],
    needle_radius_mm: float | None = None,
) -> dict[str, Any]:
    """Fit a calibrated pendant contour to a strict Young-Laplace profile."""
    obs_mm = pendant_contour_to_model_mm(
        contour_px,
        axis_x_px=axis_x_px,
        apex_y_px=apex_y_px,
        px_per_mm=px_per_mm,
    )
    if obs_mm.shape[0] < 8:
        return {
            "params": [],
            "param_names": ["r0_mm", "beta", "x_offset_mm", "z_offset_mm"],
            "residuals": {"rmse": float("nan"), "max_abs": float("nan"), "dof": 0, "r": []},
            "strict_fit_success": False,
            "strict_fit_warning": "not_enough_contour_points",
        }

    obs_mm = obs_mm[obs_mm[:, 1] >= -0.5 / float(px_per_mm)]
    envelope_mm = build_pendant_profile_envelope_mm(
        contour_px,
        axis_x_px=axis_x_px,
        apex_y_px=apex_y_px,
        px_per_mm=px_per_mm,
    )
    diameter_mm = float(np.ptp(obs_mm[:, 0]))
    height_mm = float(np.ptp(obs_mm[:, 1]))
    if envelope_mm.shape[0] >= 3:
        diameter_mm = max(diameter_mm, float(2.0 * np.max(envelope_mm[:, 0])))
        height_mm = max(height_mm, float(np.max(envelope_mm[:, 1])))
        
    x0, lower, upper = _bounds_from_seed(
        r0_seed_mm=r0_seed_mm,
        beta_seed=beta_seed,
        diameter_mm=diameter_mm,
        height_mm=height_mm,
    )

    def model_from_params(params: np.ndarray) -> np.ndarray:
        r0_mm, beta, x_offset_mm, z_offset_mm = params
        model = integrate_young_laplace_profile_mm(
            r0_mm,
            beta,
            target_height_mm=height_mm,
        )
        if model.size == 0:
            return model
        return model + np.array([x_offset_mm, z_offset_mm])

    def residuals(params: np.ndarray) -> np.ndarray:
        model = model_from_params(params)
        r = _normal_projection_residuals_mm(obs_mm, model)
        r0_mm, _beta, x_offset_mm, z_offset_mm = params
        offset_scale = max(0.05, r0_mm * 0.25)
        regularization = (
            np.array([x_offset_mm / offset_scale, z_offset_mm / offset_scale], dtype=float) * 0.01
        )
        return np.concatenate([r, regularization])

    res = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.03,
        max_nfev=300,
        method="trf",
    )

    r_vec = residuals(res.x)
    contour_r = r_vec[: obs_mm.shape[0]]
    rmse = float(np.sqrt(np.mean(contour_r**2))) if contour_r.size else float("nan")
    max_abs = float(np.max(np.abs(contour_r))) if contour_r.size else float("nan")
    model_mm = model_from_params(res.x)
    coverage_height_mm = float(np.ptp(model_mm[:, 1])) if model_mm.size else 0.0
    radial_profile_mm, radial_meta = integrate_young_laplace_profile_mm(
        float(res.x[0]),
        float(res.x[1]),
        target_height_mm=height_mm,
        branch="right",
        return_metadata=True,
    )

    rho1 = float(physics.get("rho1", 1000.0))
    rho2 = float(physics.get("rho2", 1.2))
    g = float(physics.get("g", 9.80665))
    r0_mm = float(res.x[0])
    beta = float(res.x[1])
    gamma_mn_m = float(surface_tension(rho1 - rho2, g, r0_mm, beta) * 1000.0)

    threshold_mm = max(0.05, 0.03 * max(diameter_mm, 1e-12))
    params_finite = bool(np.all(np.isfinite(res.x)))
    positive_params = bool(r0_mm > 0 and beta > 0)
    coverage_ok = bool(coverage_height_mm >= 0.98 * height_mm)
    pinned = _is_pinned(res.x, lower, upper)
    strict_success = bool(
        res.success
        and params_finite
        and positive_params
        and coverage_ok
        and not pinned
        and np.isfinite(rmse)
        and rmse <= threshold_mm
    )

    warning = None
    if not res.success:
        warning = "optimizer_failed"
    elif not params_finite or not positive_params:
        warning = "invalid_fit_parameters"
    elif pinned:
        warning = "fit_parameter_at_bound"
    elif not coverage_ok:
        warning = "model_height_coverage_failed"
    elif not np.isfinite(rmse) or rmse > threshold_mm:
        warning = "residual_gate_failed"

    model_px = model_mm_to_pendant_px(
        model_mm,
        axis_x_px=axis_x_px,
        apex_y_px=apex_y_px,
        px_per_mm=px_per_mm,
    )

    return {
        "params": res.x.tolist(),
        "param_names": ["r0_mm", "beta", "x_offset_mm", "z_offset_mm"],
        "residuals": {
            "rmse": rmse,
            "max_abs": max_abs,
            "dof": int(contour_r.size - res.x.size),
            "r": contour_r.tolist(),
            "units": "mm",
            "threshold_mm": threshold_mm,
        },
        "solver": {
            "backend": "scipy.least_squares",
            "method": "trf",
            "iterations": int(res.nfev),
            "success": bool(res.success),
            "message": str(res.message),
        },
        "strict_fit_success": strict_success,
        "strict_fit_warning": warning,
        "strict_surface_tension_mN_m": gamma_mn_m,
        "strict_r0_mm": r0_mm,
        "strict_beta": beta,
        "strict_x_offset_mm": float(res.x[2]),
        "strict_z_offset_mm": float(res.x[3]),
        "strict_rmse_mm": rmse,
        "strict_residual_threshold_mm": threshold_mm,
        "strict_fit_stop_reason": radial_meta.get("stop_reason", "unknown"),
        "strict_model_coverage_height_mm": coverage_height_mm,
        "strict_observed_height_mm": height_mm,
        "strict_observed_diameter_mm": diameter_mm,
        "observed_profile_mm": envelope_mm.tolist(),
        "model_radial_profile_mm": radial_profile_mm.tolist(),
        "model_profile_mm": model_mm.tolist(),
        "model_profile_px": model_px.tolist(),
    }

# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    sample_path = Path("data/samples/gota pendiente 1.png")
    if not sample_path.exists():
        logger.error(f"Sample image not found at {sample_path}")
        sys.exit(1)
        
    logger.info(f"Loading sample image: {sample_path}")
    image = cv2.imread(str(sample_path))
    if image is None:
        logger.error(f"Could not load image file: {sample_path}")
        sys.exit(1)
        
    # --- 1. Autocalibrate ---
    logger.info("Executing autocalibration...")
    calibration = run_pendant_autocalibration(image)
    needle_rect = calibration["needle_rect"]
    contact_pts = calibration["contact_points"]
    apex_pt = calibration["apex_point"]
    raw_contour = calibration["drop_contour"]
    
    # Calculate scale factor px_per_mm using 1.83 mm physical needle diameter
    needle_width_px = needle_rect[2]
    px_per_mm = float(needle_width_px) / NEEDLE_DIAM_MM
    logger.info(f"Autocalibration complete:")
    logger.info(f"  Needle Rect (px): x={needle_rect[0]}, y={needle_rect[1]}, w={needle_rect[2]}, h={needle_rect[3]}")
    logger.info(f"  Needle Width = {needle_width_px} px, Calibrated Scale = {px_per_mm:.4f} px/mm")
    logger.info(f"  Contact Points (px): Left={contact_pts[0]}, Right={contact_pts[1]}")
    logger.info(f"  Apex Point (px): {apex_pt}")
    
    # --- 2. Contour Extraction & Preprocessing ---
    logger.info("Executing contour extraction stage...")
    xy = _contour_to_xy(raw_contour)
    # Clip contour below needle contact level
    clipped_xy = _clip_contour_at_pendant_contacts(xy, contact_pts)
    logger.info(f"  Extracted contour: {xy.shape[0]} points -> Clipped: {clipped_xy.shape[0]} points")
    
    # --- 3. Geometric Features ---
    logger.info("Extracting geometric features...")
    axis_x = float(np.median(clipped_xy[:, 0]))
    apex_y = float(apex_pt[1])
    apex_xy = (float(apex_pt[0]), apex_y)
    logger.info(f"  Symmetry axis X: {axis_x:.2f} px")
    
    # Measure physical diameter from max width of clipped contour
    diameter_px, diameter_line = _pendant_max_width(clipped_xy, contact_pts)
    diameter_mm = diameter_px / px_per_mm
    logger.info(f"  Max Width: {diameter_px:.2f} px = {diameter_mm:.4f} mm")
    
    # Fit local circle at apex to find seed radius
    r0_seed_px = _pendant_apex_radius_px(clipped_xy, apex_xy)
    r0_seed_mm = r0_seed_px / px_per_mm if r0_seed_px > 0 else (diameter_mm / 4.0)
    logger.info(f"  Apex Radius Seed: {r0_seed_px:.2f} px = {r0_seed_mm:.4f} mm")
    
    # Jennings-Pallas seed estimation
    s1 = diameter_px / (2.0 * r0_seed_px) if r0_seed_px > 0 else 1.0
    beta_seed = jennings_pallas_beta(s1)
    logger.info(f"  Jennings-Pallas s1 (De / 2R0): {s1:.4f}, seed beta: {beta_seed:.4f}")
    
    # --- 4. Jennings-Pallas Geometric Surface Tension Estimate ---
    delta_rho = RHO1 - RHO2
    gamma_geometric = surface_tension(delta_rho, G, r0_seed_mm, beta_seed) * 1000.0
    logger.info(f"Jennings-Pallas Geometric Surface Tension: {gamma_geometric:.4f} mN/m")
    
    # --- 5. Strict Young-Laplace Profile Fitting ---
    logger.info("Executing strict Young-Laplace profile fitting solver...")
    physics = {"rho1": RHO1, "rho2": RHO2, "g": G}
    needle_radius_mm = (NEEDLE_DIAM_MM / 2.0)
    
    fit_results = fit_pendant_young_laplace_strict(
        contour_px=clipped_xy,
        axis_x_px=axis_x,
        apex_y_px=apex_y,
        px_per_mm=px_per_mm,
        r0_seed_mm=r0_seed_mm,
        beta_seed=beta_seed,
        physics=physics,
        needle_radius_mm=needle_radius_mm,
    )
    
    # --- 6. Derived Metric Calculations ---
    logger.info("Calculating derived metrics...")
    
    # Obtain profile for integrals (use fitted model or fallback envelope)
    profile_for_integrals = None
    envelope_mm = build_pendant_profile_envelope_mm(
        clipped_xy,
        axis_x_px=axis_x,
        apex_y_px=apex_y,
        px_per_mm=px_per_mm,
    )
    
    if fit_results.get("strict_fit_success") and fit_results.get("model_radial_profile_mm"):
        profile_for_integrals = np.asarray(fit_results["model_radial_profile_mm"])
        r0_mm = fit_results["strict_r0_mm"]
        beta = fit_results["strict_beta"]
        gamma_mN_m = fit_results["strict_surface_tension_mN_m"]
        method_str = "young_laplace_strict"
    else:
        profile_for_integrals = envelope_mm
        r0_mm = r0_seed_mm
        beta = beta_seed
        gamma_mN_m = gamma_geometric
        method_str = "jennings_pallas_geometric"
        
    volume_uL, drop_surface_mm2 = _radial_profile_integrals(profile_for_integrals)
    
    # Bond and Worthington numbers
    gamma_N_m = gamma_mN_m / 1000.0
    bond_num = bond_number(delta_rho, G, r0_mm, gamma_N_m)
    vmax = vmax_uL(gamma_N_m, NEEDLE_DIAM_MM, delta_rho, G)
    worth_num = worthington_number(volume_uL, vmax)
    
    # Print beautiful reports
    print("\n" + "="*80)
    print(" MENIPY PENDANT DROP PIPELINE BASELINE REPORT")
    print("="*80)
    print(f"Image analyzed:              {sample_path.name}")
    print(f"Symmetry Axis (px):          {axis_x:.2f}")
    print(f"Apex Location (px):          {apex_xy}")
    print(f"Scale factor (px/mm):        {px_per_mm:.4f}")
    print(f"Detached Maximum Width (mm): {diameter_mm:.4f}")
    print("-"*80)
    print(" METRICS COMPARISON TABLE")
    print("-"*80)
    print(f" {'Metric':<30} | {'Jennings-Pallas (Geom)':<22} | {'Young-Laplace (Strict)':<22}")
    print(f" {'-'*30} | {'-'*22} | {'-'*22}")
    print(f" {'Method':<30} | {'jennings_pallas_geometric':<22} | {'young_laplace_strict':<22}")
    
    status_str = "SUCCESS" if fit_results["strict_fit_success"] else f"FAILED ({fit_results['strict_fit_warning']})"
    print(f" {'Solver Status':<30} | {'SUCCESS':<22} | {status_str:<22}")
    print(f" {'Apex curvature radius R0 (mm)':<30} | {r0_seed_mm:<22.4f} | {fit_results.get('strict_r0_mm', float('nan')):<22.4f}")
    print(f" {'Form factor beta':<30} | {beta_seed:<22.4f} | {fit_results.get('strict_beta', float('nan')):<22.4f}")
    print(f" {'Surface Tension (mN/m)':<30} | {gamma_geometric:<22.4f} | {fit_results.get('strict_surface_tension_mN_m', float('nan')):<22.4f}")
    print(f" {'Fit Residual RMSE (mm)':<30} | {'N/A':<22} | {fit_results.get('strict_rmse_mm', float('nan')):<22.4f}")
    
    print("-"*80)
    print(" REPORTED DERIVED VALUES (Based on chosen solver)")
    print("-"*80)
    print(f" Chosen Solver:              {method_str}")
    print(f" Droplet Volume (uL):        {volume_uL:.4f} (Expected: ~7.92 uL)")
    print(f" Surface Area (mm2):         {drop_surface_mm2:.4f} (Expected: ~17.27 mm2)")
    print(f" Bond number:                {bond_num:.4f} (Expected: ~0.388)")
    print(f" Worthington number:         {worth_num:.4f} (Expected: ~0.462)")
    print(f" Max volume V_max (uL):      {vmax:.4f}")
    print("="*80 + "\n")
    
    # --- 7. Visualization Overlay and Saving ---
    logger.info("Generating visual overlay...")
    height, width = image.shape[:2]
    annotated = image.copy()
    
    # 1. Draw raw clipped contour (yellow)
    pts_px = clipped_xy.astype(int)
    for p in pts_px:
        cv2.circle(annotated, tuple(p), 1, (0, 255, 255), -1) # yellow dots
        
    # 2. Draw axis of symmetry (cyan line)
    ax_x = int(round(axis_x))
    cv2.line(annotated, (ax_x, 0), (ax_x, height), (255, 255, 0), 1) # cyan line
    
    # 3. Draw apex (red cross)
    cv2.drawMarker(annotated, apex_pt, (0, 0, 255), cv2.MARKER_CROSS, 10, 2) # red cross
    
    # 4. Draw fitted strict Young-Laplace profile (green curve)
    if fit_results.get("strict_fit_success") and "model_profile_px" in fit_results:
        model_px = np.asarray(fit_results["model_profile_px"], dtype=int)
        for i in range(len(model_px) - 1):
            cv2.line(annotated, tuple(model_px[i]), tuple(model_px[i+1]), (0, 255, 0), 2) # green line
            
    # 5. Draw text results
    text_st = f"IFT: {gamma_mN_m:.2f} mN/m ({method_str})"
    text_vol = f"Vol: {volume_uL:.2f} uL, R0: {r0_mm:.2f} mm"
    cv2.putText(annotated, text_st, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated, text_vol, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    output_path = Path("gota_pendiente_baseline.png")
    cv2.imwrite(str(output_path), annotated)
    logger.info(f"Annotated result image saved to: {output_path.resolve()}")
    
    # 6. Save coordinate data to text file
    txt_path = Path("gota_pendiente_coords.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Menipy Drop Analysis Coordinates\n")
        f.write(f"# Image: {sample_path.name}\n")
        f.write(f"# Scale: {px_per_mm:.6f} px/mm\n\n")
        
        f.write("# NEEDLE BBOX (pixel space: x, y, width, height)\n")
        f.write(f"{needle_rect[0]}, {needle_rect[1]}, {needle_rect[2]}, {needle_rect[3]}\n\n")
        
        f.write("# NEEDLE CONTACT POINTS (pixel space: Left_x, Left_y, Right_x, Right_y)\n")
        f.write(f"{contact_pts[0][0]}, {contact_pts[0][1]}, {contact_pts[1][0]}, {contact_pts[1][1]}\n\n")
        
        f.write("# DROPLET CONTOUR COORDINATES (pixel space: x, y)\n")
        for pt in clipped_xy:
            f.write(f"{pt[0]:.2f}, {pt[1]:.2f}\n")
        f.write("\n")
        
        f.write("# PREDICTED CONTOUR COORDINATES (pixel space: x, y)\n")
        if fit_results.get("strict_fit_success") and "model_profile_px" in fit_results:
            model_px = np.asarray(fit_results["model_profile_px"])
            for pt in model_px:
                f.write(f"{pt[0]:.2f}, {pt[1]:.2f}\n")
        else:
            f.write("# (Fitted Young-Laplace profile not available)\n")
            
    logger.info(f"Coordinates saved to: {txt_path.resolve()}")
    
    # Self-check assertions matching pytest regression bounds
    assert abs(volume_uL - 7.92) < 0.8, f"Volume {volume_uL:.3f} deviated too much from ~7.92"
    assert abs(gamma_mN_m - 29.24) < 6.0, f"Surface tension {gamma_mN_m:.3f} deviated too much from ~29.24"
    logger.info("Verification checks PASSED successfully.")

if __name__ == "__main__":
    main()
