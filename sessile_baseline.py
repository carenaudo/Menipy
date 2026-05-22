#!/usr/bin/env python3
"""
Sessile Drop Autocalibration and Strict Young-Laplace Fitting Baseline Script.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sessile_baseline")

# Physical constants matching Menipy defaults
RHO1 = 1000.0        # drop density (water) in kg/m3
RHO2 = 1.2           # fluid/phase density (air) in kg/m3
G = 9.80665          # gravity in m/s2
NEEDLE_DIAM_MM = 1.83 # physical needle diameter in mm


def cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the 2D cross product (determinant) of a and b.

    Supports 1D vs 1D, 1D vs 2D, and 2D vs 1D inputs.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        return a[0] * b[1] - a[1] * b[0]
    elif a.ndim == 1 and b.ndim == 2:
        return a[0] * b[:, 1] - a[1] * b[:, 0]
    elif a.ndim == 2 and b.ndim == 1:
        return a[:, 0] * b[1] - a[:, 1] * b[0]
    else:
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


# ==============================================================================
# 1. GEOMETRIC & PHYSICAL HELPER FUNCTIONS (From Menipy common & models)
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


def volume_from_contour(contour_mm: np.ndarray) -> float:
    """Return droplet volume in microlitres from a 2-D contour.

    Parameters
    ----------
    contour_mm:
        ``Nx2`` array of ``(r, y)`` coordinates in millimetres, where ``r`` is
        the radial distance from the symmetry axis and ``y`` is the vertical
        coordinate.
    """
    r = contour_mm[:, 0].astype(float) / 1000.0
    y = contour_mm[:, 1].astype(float) / 1000.0
    idx = np.argsort(y)
    y_sorted = y[idx]
    r_sorted = r[idx]
    trapz_val = trapezoid(r_sorted**2, y_sorted)
    vol = float(np.pi) * float(trapz_val)  # m^3
    return float(vol * 1e9)


def surface_area_mm2(contour_px: np.ndarray, px_per_mm: float) -> float:
    """Surface area of a revolved contour (mm²)."""
    contour_px = np.asarray(contour_px, dtype=float)
    r_mm = contour_px[:, 0] / px_per_mm
    z_mm = contour_px[:, 1] / px_per_mm
    order = np.argsort(z_mm)
    r_mm, z_mm = r_mm[order], z_mm[order]
    z_mm, unique_idx = np.unique(z_mm, return_index=True)
    r_mm = r_mm[unique_idx].astype(float)

    if len(r_mm) < 3 or len(z_mm) < 3:
        return 0.0

    dr_dz = np.gradient(r_mm, z_mm, edge_order=2)
    integrand = np.asarray(r_mm * np.sqrt(1.0 + dr_dz**2), dtype=float)
    trapz_val = float(trapezoid(integrand, z_mm))
    A_mm2 = 2.0 * math.pi * trapz_val
    return float(A_mm2)


def _substrate_frame(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return substrate tangent and apex-side normal unit vectors."""
    p1 = np.asarray(substrate_line[0], dtype=float)
    p2 = np.asarray(substrate_line[1], dtype=float)
    substrate_vec = p2 - p1
    norm = np.linalg.norm(substrate_vec)
    if norm <= 0:
        substrate_vec = np.array([1.0, 0.0], dtype=float)
    else:
        substrate_vec = substrate_vec / norm

    normal_vec = np.array([-substrate_vec[1], substrate_vec[0]], dtype=float)
    rel = np.asarray(contour, dtype=float).reshape(-1, 2) - np.asarray(
        contact_point, dtype=float
    )
    signed_heights = rel @ normal_vec
    nonzero = signed_heights[np.abs(signed_heights) > 0.5]
    if nonzero.size and float(np.median(nonzero)) < 0:
        normal_vec = -normal_vec
    return substrate_vec, normal_vec


def _contact_branch_points(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    window_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select apex-side inward-branch points near a sessile contact point."""
    contour_2d = np.asarray(contour, dtype=float).reshape(-1, 2)
    contact = np.asarray(contact_point, dtype=float)
    substrate_vec, normal_vec = _substrate_frame(contour_2d, contact, substrate_line)

    rel_all = contour_2d - contact
    s_all = rel_all @ substrate_vec
    h_all = rel_all @ normal_vec
    apex_side = h_all > 0.5
    if np.any(apex_side):
        inward_sign = 1.0 if float(np.median(s_all[apex_side])) >= 0 else -1.0
    else:
        inward_sign = 1.0

    best_points = np.empty((0, 2), dtype=float)
    for factor in (1.0, 1.5, 2.0, 3.0):
        radius = float(window_px) * factor
        distances = np.linalg.norm(rel_all, axis=1)
        mask = (distances <= radius) & (h_all > 0.5) & ((s_all * inward_sign) >= -1.0)
        local_points = contour_2d[mask]
        if local_points.shape[0] >= 3:
            return local_points, substrate_vec, normal_vec
        if local_points.shape[0] > best_points.shape[0]:
            best_points = local_points

    return best_points, substrate_vec, normal_vec


def estimate_contact_angle_tangent(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    window_px: int = 15,
    method: str = "poly",
) -> tuple[float, float]:
    """Estimate contact angle using tangent method near contact point."""
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")

    if method == "poly":
        try:
            local_points, substrate_vec, normal_vec = _contact_branch_points(
                contour, contact_point, substrate_line, window_px
            )
            if len(local_points) < 3:
                return 90.0, 1.0

            vectors = local_points - np.asarray(contact_point, dtype=float)
            distances = np.linalg.norm(vectors, axis=1)
            weights = 1.0 / np.maximum(distances, 1.0) ** 2
            _, _, vh = np.linalg.svd(
                vectors * np.sqrt(weights[:, None]), full_matrices=False
            )
            tangent = vh[0]
            tangent /= np.linalg.norm(tangent) or 1.0
            along = abs(float(np.dot(tangent, substrate_vec)))
            upward = abs(float(np.dot(tangent, normal_vec)))
            angle_deg = float(np.degrees(np.arctan2(upward, along)))
            distances_to_line = np.abs(
                vectors[:, 0] * tangent[1] - vectors[:, 1] * tangent[0]
            )
            rmse = float(np.sqrt(np.average(distances_to_line**2, weights=weights)))
        except Exception:
            angle_deg = 90.0
            rmse = 1.0
    elif method == "arc":
        try:
            local_points, substrate_vec, _normal_vec = _contact_branch_points(
                contour, contact_point, substrate_line, window_px
            )
            if len(local_points) < 5:
                return 90.0, 1.0

            center, radius = fit_circle(local_points)
            vec_to_contact = contact_point - center
            dist = np.linalg.norm(vec_to_contact)
            if dist > max(1e-9, radius * 1e-9):
                radial = vec_to_contact / dist
                tangent = np.array([-radial[1], radial[0]], dtype=float)
                along = abs(float(np.dot(tangent, substrate_vec)))
                upward = np.sqrt(max(0.0, 1.0 - along**2))
                angle_deg = float(np.degrees(np.arctan2(upward, along)))
            else:
                angle_deg = 90.0
            distances_to_circle = np.abs(
                np.linalg.norm(local_points - center, axis=1) - radius
            )
            rmse = float(np.sqrt(np.mean(distances_to_circle**2)))
        except Exception:
            angle_deg = 90.0
            rmse = 1.0
    else:
        angle_deg = 90.0
        rmse = 1.0

    return angle_deg, rmse


def estimate_contact_angle_circle_fit(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    window_px: int = 30,
) -> tuple[float, float]:
    """Estimate contact angle using circle fit method."""
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be of shape (N, 2)")

    try:
        local_points, substrate_vec, _normal_vec = _contact_branch_points(
            contour, contact_point, substrate_line, window_px
        )
        if len(local_points) < 5:
            return 90.0, 1.0

        center, radius = fit_circle(local_points)
        vec_to_contact = contact_point - center
        dist = np.linalg.norm(vec_to_contact)
        if dist > max(1e-9, radius * 1e-9):
            radial = vec_to_contact / dist
            tangent = np.array([-radial[1], radial[0]], dtype=float)
            along = abs(float(np.dot(tangent, substrate_vec)))
            upward = np.sqrt(max(0.0, 1.0 - along**2))
            angle_deg = float(np.degrees(np.arctan2(upward, along)))
        else:
            angle_deg = 90.0

        distances_to_circle = np.abs(
            np.linalg.norm(local_points - center, axis=1) - radius
        )
        rmse = np.sqrt(np.mean(distances_to_circle**2))
    except Exception:
        angle_deg = 90.0
        rmse = 1.0

    return angle_deg, rmse


def tangent_angle_at_point(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    window_px: int = 15,
) -> tuple[float, float]:
    """Estimate contact angle at a point using local tangent (polynomial fit)."""
    return estimate_contact_angle_tangent(
        contour, contact_point, substrate_line, window_px, method="poly"
    )


def circle_fit_angle_at_point(
    contour: np.ndarray,
    contact_point: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]],
    window_px: int = 30,
) -> tuple[float, float]:
    """Estimate contact angle at a point using local circle fit."""
    return estimate_contact_angle_circle_fit(
        contour, contact_point, substrate_line, window_px
    )

# ==============================================================================
# 2. AUTOCALIBRATION & DETECTIONS (From Menipy common & preprocessors)
# ==============================================================================

def _find_horizon_median(strip_gray: np.ndarray) -> Optional[int]:
    """Find horizon line in a vertical strip using gradient analysis."""
    detected_ys: List[int] = []
    h, w = strip_gray.shape
    min_limit, max_limit = int(h * 0.05), int(h * 0.95)

    for col in range(w):
        col_data = strip_gray[:, col].astype(float)
        grad = np.diff(col_data)
        valid_grad = grad[min_limit:max_limit]

        if len(valid_grad) == 0:
            continue

        best_idx = np.argmin(valid_grad)
        best_y = best_idx + min_limit
        detected_ys.append(best_y)

    if not detected_ys:
        return None

    return int(np.median(detected_ys))


def run_sessile_autocalibration(image: np.ndarray) -> dict:
    """
    Perform automatic segmentation and region/substrate/needle/contour detection
    for a sessile drop image, matching AutoCalibrator._detect_sessile.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    height, width = gray.shape[:2]
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # 1. Detect substrate baseline
    margin_px = max(10, min(50, int(width * 0.05)))
    left_strip = enhanced_gray[:, 0:margin_px]
    right_strip = enhanced_gray[:, width - margin_px : width]

    y_left = _find_horizon_median(left_strip)
    y_right = _find_horizon_median(right_strip)

    if y_left is None and y_right is None:
        substrate_y = int(height * 0.8)
    else:
        if y_left is None:
            y_left = y_right
        if y_right is None:
            y_right = y_left
        substrate_y = int((y_left + y_right) / 2)
        
    substrate_line = ((0, substrate_y), (width, substrate_y))
    
    # 2. Segment using adaptive thresholding
    blur = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, # adaptive_block_size
        2,  # adaptive_c
    )
    
    # Mask below substrate line
    binary[substrate_y - 2 :, :] = 0

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Detect needle (contour touching the top border)
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    needle_rect = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y < 5:
            needle_rect = (x, y, w, h)
            break
            
    # 4. Detect drop contour
    center_x = width // 2
    min_area = height * width * 0.005
    substrate_touch_tolerance = 15

    substrate_contours = []
    floating_contours = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        top_y = y
        bottom_y = y + h

        if area < min_area:
            continue

        if x < 5 or (x + w) > (width - 5):
            continue

        if needle_rect is not None:
            n_x, n_y, n_w, n_h = needle_rect
            needle_bottom = n_y + n_h
            needle_center_x = n_x + n_w // 2

            min_gap_from_needle = 50
            if top_y < needle_bottom + min_gap_from_needle:
                continue

            cnt_center_x = x + w // 2
            if (
                abs(cnt_center_x - needle_center_x) < n_w
                and top_y < needle_bottom + 100
            ):
                continue

        rect_area = w * h
        if rect_area > 0:
            rectangularity = area / rect_area
            if rectangularity > 0.85:
                continue

        if substrate_y is not None:
            cnt_cy = y + h // 2
            if cnt_cy > substrate_y:
                continue

        cnt_center_x = x + w // 2
        distance_from_center = abs(cnt_center_x - center_x)

        if substrate_y is not None:
            distance_to_substrate = abs(bottom_y - substrate_y)
            touches_substrate = distance_to_substrate <= substrate_touch_tolerance

            if touches_substrate:
                substrate_contours.append(
                    (cnt, area, distance_from_center, distance_to_substrate)
                )
            else:
                floating_contours.append(
                    (cnt, area, distance_from_center, distance_to_substrate)
                )
        else:
            floating_contours.append((cnt, area, distance_from_center, 0))

    if substrate_contours:
        substrate_contours.sort(key=lambda x: (x[3], -x[1], x[2]))
        best_cnt = substrate_contours[0][0]
    elif floating_contours:
        floating_contours.sort(key=lambda x: (-x[1], x[2]))
        best_cnt = floating_contours[0][0]
    else:
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > min_area:
                best_cnt = largest
            else:
                raise ValueError("No valid sessile drop contour found.")
        else:
            raise ValueError("No valid sessile drop contour found.")

    # Extract contact points and build final_polygon from best_cnt
    hull = cv2.convexHull(best_cnt)
    points = hull[:, 0, :]

    substrate_tolerance = 20
    near_substrate = [
        pt
        for pt in points
        if abs(pt[1] - substrate_y) <= substrate_tolerance
    ]
    dome_raw = [
        pt
        for pt in points
        if pt[1] <= (substrate_y + substrate_tolerance)
    ]
    dome_points = []
    for pt in dome_raw:
        if pt[1] > substrate_y:
            dome_points.append(
                np.array([pt[0], substrate_y], dtype=pt.dtype)
            )
        else:
            dome_points.append(pt)

    if near_substrate:
        sorted_near = sorted(near_substrate, key=lambda p: p[0])
        x_left = sorted_near[0][0]
        x_right = sorted_near[-1][0]
    elif len(points) > 0:
        sorted_pts = sorted(points, key=lambda p: p[0])
        x_left = sorted_pts[0][0]
        x_right = sorted_pts[-1][0]
    else:
        raise ValueError("No points found in convex hull")

    cp_left = (int(x_left), substrate_y)
    cp_right = (int(x_right), substrate_y)
    contact_points = (cp_left, cp_right)

    if dome_points:
        dome_points = sorted(dome_points, key=lambda p: p[0])
        final_polygon = np.array(
            [[x_left, substrate_y]]
            + [[p[0], p[1]] for p in dome_points]
            + [[x_right, substrate_y]]
            + [[x_left, substrate_y]],
            dtype=np.float64,
        )
    else:
        all_pts = sorted(points, key=lambda p: p[0])
        final_polygon = np.array(
            [[x_left, substrate_y]]
            + [[p[0], p[1]] for p in all_pts]
            + [[x_right, substrate_y]]
            + [[x_left, substrate_y]],
            dtype=np.float64,
        )
        
    return {
        "drop_contour": final_polygon,
        "substrate_line": substrate_line,
        "substrate_y": substrate_y,
        "needle_rect": needle_rect,
        "contact_points": contact_points,
        "binary_mask": binary_clean,
        "enhanced_image": enhanced_gray
    }

# ==============================================================================
# 3. YOUNG-LAPLACE SOLVER & OPTIMIZER (From Menipy math & common)
# ==============================================================================

def young_laplace_ode(
    params: np.ndarray,
    physics: Dict[str, Any],
) -> np.ndarray:
    """
    Integrate the axisymmetric Young-Laplace ODE using Bashforth-Adams formulation.
    Returns (N, 2) array of [r, z] profile coordinates in mm.
    """
    R0_mm = float(params[0])
    beta = float(params[1])

    if R0_mm <= 0:
        return np.array([[0.0, 0.0]])

    def odesys(s, y):
        # y = [r, z, psi]
        r, z, psi = y

        if r < 1e-12:
            sin_psi_r = 1.0 / R0_mm
        else:
            sin_psi_r = np.sin(psi) / r

        drds = np.cos(psi)
        dzds = np.sin(psi)
        dpsids = (2.0 / R0_mm) + (beta / (R0_mm**2)) * z - sin_psi_r

        return [drds, dzds, dpsids]

    def hit_axis(s, y):
        return y[0] - 1e-6 if s > 0.1 else 1.0

    hit_axis.terminal = True
    hit_axis.direction = -1

    # Max arc length to integrate
    s_max = 5.0 * R0_mm * max(1.0, 1.0 / max(1e-3, abs(beta)))
    y0 = [0.0, 0.0, 0.0]

    sol = solve_ivp(
        odesys,
        [0.0, s_max],
        y0,
        method="RK45",
        events=hit_axis,
        max_step=s_max / 200.0,
        rtol=1e-5,
        atol=1e-6,
    )

    r_right = sol.y[0]
    z_right = sol.y[1]

    # Mirror for full silhouette
    r_left = -r_right[::-1]
    z_left = z_right[::-1]

    r_full = np.concatenate([r_left[:-1], r_right])
    z_full = np.concatenate([z_left[:-1], z_right])

    return np.column_stack([r_full, z_full])


def fit_sessile_young_laplace(
    contour_px: np.ndarray,
    x0: list[float] = [20.0, 0.1],
    bounds: tuple[list[float], list[float]] = ([1.0, -10.0], [2000.0, 10.0]),
) -> dict:
    """Fit a calibrated/uncalibrated sessile contour to a Young-Laplace profile."""
    obs_xy = np.asarray(contour_px, dtype=float)

    def _residuals_pointwise(obs_xy: np.ndarray, model_xy: np.ndarray) -> np.ndarray:
        """Pointwise residual: pair points by normalized arc-length."""
        def _arclen_param(xy):
            seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            return s / (s[-1] if s[-1] > 0 else 1.0)

        def _resample(xy, u, m=400):
            v = _arclen_param(xy)
            xs = np.interp(u, v, xy[:, 0])
            ys = np.interp(u, v, xy[:, 1])
            return np.column_stack([xs, ys])

        m = min(len(obs_xy), 400)
        u = np.linspace(0.0, 1.0, m)
        obs_r = _resample(obs_xy, u, m=m)
        mod_r = _resample(model_xy, u, m=m)
        diff = obs_r - mod_r
        return diff.reshape(-1)

    def fun(x: np.ndarray) -> np.ndarray:
        # Integrator returns profile in mm, which are compared directly to obs_xy (px)
        model_xy = young_laplace_ode(x, {})
        return _residuals_pointwise(obs_xy, model_xy)

    res = least_squares(
        fun,
        x0=np.asarray(x0, dtype=float),
        bounds=(
            np.asarray(bounds[0], dtype=float),
            np.asarray(bounds[1], dtype=float),
        ),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=300,
        method="trf",
    )

    r_vec = res.fun
    rmse = float(np.sqrt(np.mean(r_vec**2))) if r_vec.size else float("nan")
    max_abs = float(np.max(np.abs(r_vec))) if r_vec.size else float("nan")

    return {
        "params": res.x.tolist(),
        "residuals": {
            "rmse": rmse,
            "max_abs": max_abs,
            "dof": int(r_vec.size - res.x.size),
        },
        "solver": {
            "backend": "scipy.least_squares",
            "method": "trf",
            "iterations": int(res.nfev),
            "success": bool(res.success),
            "message": str(res.message),
        }
    }

# ==============================================================================
# 4. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    sample_path = Path("data/samples/gota depositada 1.png")
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
    calibration = run_sessile_autocalibration(image)
    needle_rect = calibration["needle_rect"]
    substrate_line = calibration["substrate_line"]
    substrate_y = calibration["substrate_y"]
    contact_pts = calibration["contact_points"]
    final_polygon = calibration["drop_contour"]

    # Calculate scale factor px_per_mm using 1.83 mm physical needle diameter
    needle_width_px = needle_rect[2]
    px_per_mm = float(needle_width_px) / NEEDLE_DIAM_MM
    logger.info(f"Autocalibration complete:")
    logger.info(f"  Needle Rect (px): x={needle_rect[0]}, y={needle_rect[1]}, w={needle_rect[2]}, h={needle_rect[3]}")
    logger.info(f"  Needle Width = {needle_width_px} px, Calibrated Scale = {px_per_mm:.4f} px/mm")
    logger.info(f"  Substrate Line (px): {substrate_line}")
    logger.info(f"  Contact Points (px): Left={contact_pts[0]}, Right={contact_pts[1]}")

    # --- 2. Determine Apex ---
    logger.info("Locating apex point...")
    # Matches pipeline logic: min y on final_polygon.
    # Since np.argmin returns first index of minimum, and y has its minimum first at [231, 178], it matches (231, 178).
    y_coords = final_polygon[:, 1]
    apex_i = int(np.argmin(y_coords))
    apex_xy = (float(final_polygon[apex_i, 0]), float(final_polygon[apex_i, 1]))
    logger.info(f"  Apex Point (px): {apex_xy}")

    # --- 3. Geometric Features & Derived Metrics ---
    logger.info("Extracting geometric features and derived metrics...")
    
    # Calculate height and tilt-corrected diameter
    p1_line = np.array(substrate_line[0])
    p2_line = np.array(substrate_line[1])
    apex_pt = np.array(apex_xy)
    
    # Distance from point to line
    num = float(np.abs(cross_2d(p2_line - p1_line, p1_line - apex_pt)))
    den = float(np.linalg.norm(p2_line - p1_line))
    height_px = num / den if den > 0 else 0.0

    p1 = np.array(contact_pts[0], dtype=float)
    p2 = np.array(contact_pts[1], dtype=float)
    line_vec = p2_line - p1_line
    line_len = np.linalg.norm(line_vec)
    if line_len > 0:
        unit_line = line_vec / line_len
        p1_proj = float(np.dot(p1 - p1_line, unit_line))
        p2_proj = float(np.dot(p2 - p1_line, unit_line))
        diameter_px = float(abs(p2_proj - p1_proj))
    else:
        diameter_px = float(np.linalg.norm(p1 - p2))

    diameter_mm = diameter_px / px_per_mm
    height_mm = height_px / px_per_mm
    
    # Base area (contact surface area)
    base_radius_mm = diameter_mm / 2.0
    contact_surface_mm2 = np.pi * (base_radius_mm**2)

    # Volume and surface area by solid of revolution
    volume_uL = 0.0
    drop_surface_mm2 = 0.0
    
    p1_sub, p2_sub = np.array(contact_pts[0]), np.array(contact_pts[1])
    v_sub = p2_sub - p1_sub
    v_axis = np.array([-v_sub[1], v_sub[0]])
    v_axis = v_axis / (np.linalg.norm(v_axis) or 1)
    
    side = np.sign(cross_2d(v_sub, apex_pt - p1_sub))
    profile_mask = np.sign(cross_2d(v_sub, final_polygon - p1_sub)) == side
    profile = final_polygon[profile_mask]

    if profile.size > 0 and len(profile) > 3:
        vec_pa = profile - apex_pt
        z_coords_px = np.dot(vec_pa, v_axis)
        r_coords_px = np.abs(cross_2d(vec_pa, v_axis))
        contour_mm = np.column_stack([r_coords_px, z_coords_px]) / px_per_mm
        volume_uL = volume_from_contour(contour_mm)
        drop_surface_mm2 = surface_area_mm2(contour_mm * px_per_mm, px_per_mm)

    # Calculate local contact angles using polynomial SVD tangent
    theta_left_deg, uncertainty_left = tangent_angle_at_point(final_polygon, contact_pts[0], substrate_line)
    theta_right_deg, uncertainty_right = tangent_angle_at_point(final_polygon, contact_pts[1], substrate_line)
    contact_angle_deg = (theta_left_deg + theta_right_deg) / 2.0

    # Calculate circle fit contact angles for comparison
    theta_left_circle, uncertainty_left_circle = circle_fit_angle_at_point(final_polygon, contact_pts[0], substrate_line)
    theta_right_circle, uncertainty_right_circle = circle_fit_angle_at_point(final_polygon, contact_pts[1], substrate_line)
    contact_angle_circle_deg = (theta_left_circle + theta_right_circle) / 2.0

    # --- 4. Young-Laplace Profile Fitting ---
    logger.info("Executing strict Young-Laplace profile fitting solver...")
    fit_results = fit_sessile_young_laplace(final_polygon)
    
    fit_R0_mm = fit_results["params"][0]
    fit_beta = fit_results["params"][1]
    rmse_val = fit_results["residuals"]["rmse"]
    
    status_str = "SUCCESS" if fit_results["solver"]["success"] else "FAILED"
    warning_str = "profile_fit_unreliable" if rmse_val > 25.0 else "None"

    # Print beautiful reports
    print("\n" + "="*80)
    print(" MENIPY SESSILE DROP PIPELINE BASELINE REPORT")
    print("="*80)
    print(f"Image analyzed:              {sample_path.name}")
    print(f"Symmetry Axis (px):          {np.median(final_polygon[:, 0]):.2f}")
    print(f"Apex Location (px):          ({apex_xy[0]:.1f}, {apex_xy[1]:.1f})")
    print(f"Scale factor (px/mm):        {px_per_mm:.4f}")
    print(f"Base Diameter (mm):          {diameter_mm:.4f}")
    print(f"Height (mm):                 {height_mm:.4f}")
    print("-"*80)
    print(" METRICS COMPARISON TABLE")
    print("-"*80)
    print(f" {'Metric':<30} | {'Calculated baseline':<22} | {'Reference Expected':<22}")
    print(f" {'-'*30} | {'-'*22} | {'-'*22}")
    print(f" {'Solver Status':<30} | {status_str:<22} | {'SUCCESS':<22}")
    print(f" {'Fit Warning':<30} | {warning_str:<22} | {'profile_fit_unreliable':<22}")
    print(f" {'Apex curvature R0 (mm)':<30} | {fit_R0_mm:<22.4f} | {17.6229:<22.4f}")
    print(f" {'Form factor beta':<30} | {fit_beta:<22.4f} | {0.0958:<22.4f}")
    print(f" {'Fit Residual RMSE (px)':<30} | {rmse_val:<22.4f} | {182.7605:<22.4f}")
    print(f" {'Contact Angle - Left (deg)':<30} | {theta_left_deg:<22.4f} | {85.9954:<22.4f}")
    print(f" {'Contact Angle - Right (deg)':<30} | {theta_right_deg:<22.4f} | {82.5220:<22.4f}")
    print(f" {'Contact Angle - Avg (deg)':<30} | {contact_angle_deg:<22.4f} | {84.2587:<22.4f}")
    print(f" {'Circle Fit Angle - Avg (deg)':<30} | {contact_angle_circle_deg:<22.4f} | {'~84.4':<22}")
    print(f" {'Volume (uL)':<30} | {volume_uL:<22.4f} | {67.1475:<22.4f}")
    print(f" {'Drop Surface Area (mm2)':<30} | {drop_surface_mm2:<22.4f} | {314.8126:<22.4f}")
    print(f" {'Base Contact Area (mm2)':<30} | {contact_surface_mm2:<22.4f} | {47.0794:<22.4f}")
    print("="*80 + "\n")

    # --- 5. Visualization Overlay and Saving ---
    logger.info("Generating visual overlay...")
    height, width = image.shape[:2]
    annotated = image.copy()

    # 1. Draw raw contour (green polyline)
    pts_px = final_polygon.astype(int)
    # Recreate the points array for cv2.polylines
    cv2.polylines(annotated, [pts_px], isClosed=True, color=(0, 255, 0), thickness=2)

    # 2. Draw substrate baseline (yellow line)
    cv2.line(annotated, substrate_line[0], substrate_line[1], (0, 255, 255), 1)

    # 3. Draw axis of symmetry (cyan line)
    ax_x = int(round(np.median(final_polygon[:, 0])))
    cv2.line(annotated, (ax_x, 0), (ax_x, height), (255, 255, 0), 1)

    # 4. Draw apex (blue dot)
    cv2.circle(annotated, (int(apex_xy[0]), int(apex_xy[1])), 5, (255, 0, 0), -1)

    # 5. Draw fitted Young-Laplace profile (red curve)
    # Generate integrated Young-Laplace profile in mm
    model_mm = young_laplace_ode([fit_R0_mm, fit_beta], {})
    # Map back to pixel coordinates
    # X_px = r * px_per_mm + apex_x
    # Y_px = z * px_per_mm + apex_y
    fit_x_px = model_mm[:, 0] * px_per_mm + apex_xy[0]
    fit_y_px = model_mm[:, 1] * px_per_mm + apex_xy[1]
    
    # Clip coordinates below substrate
    valid_mask = fit_y_px <= (substrate_y + 1.0)
    fit_x_px = fit_x_px[valid_mask]
    fit_y_px = fit_y_px[valid_mask]
    
    model_px = np.column_stack([fit_x_px, fit_y_px]).astype(int)
    for i in range(len(model_px) - 1):
        cv2.line(annotated, tuple(model_px[i]), tuple(model_px[i+1]), (0, 0, 255), 2) # red line

    # 6. Draw contact points (red circles with small crosses)
    for cp in contact_pts:
        cv2.circle(annotated, cp, 4, (0, 0, 255), 1)
        cv2.drawMarker(annotated, cp, (0, 0, 255), cv2.MARKER_CROSS, 8, 1)

    # 7. Draw text results
    text_angles = f"Theta: L={theta_left_deg:.1f}deg, R={theta_right_deg:.1f}deg, Avg={contact_angle_deg:.1f}deg"
    text_st = f"Vol: {volume_uL:.1f} uL, Area: {drop_surface_mm2:.1f} mm2"
    text_fit = f"YL params: R0={fit_R0_mm:.2f} mm, beta={fit_beta:.4f}"
    
    cv2.putText(annotated, text_angles, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(annotated, text_st, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(annotated, text_fit, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    output_path = Path("gota_depositada_baseline.png")
    cv2.imwrite(str(output_path), annotated)
    logger.info(f"Annotated result image saved to: {output_path.resolve()}")

    # --- 6. Save Coordinates Data to Text File ---
    txt_path = Path("gota_depositada_coords.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# Menipy Drop Analysis Coordinates\n")
        f.write(f"# Image: {sample_path.name}\n")
        f.write(f"# Scale: {px_per_mm:.6f} px/mm\n\n")

        f.write("# NEEDLE BBOX (pixel space: x, y, width, height)\n")
        f.write(f"{needle_rect[0]}, {needle_rect[1]}, {needle_rect[2]}, {needle_rect[3]}\n\n")

        f.write("# SUBSTRATE BASELINE (pixel space: x1, y1, x2, y2)\n")
        f.write(f"{substrate_line[0][0]}, {substrate_line[0][1]}, {substrate_line[1][0]}, {substrate_line[1][1]}\n\n")

        f.write("# CONTACT POINTS (pixel space: Left_x, Left_y, Right_x, Right_y)\n")
        f.write(f"{contact_pts[0][0]}, {contact_pts[0][1]}, {contact_pts[1][0]}, {contact_pts[1][1]}\n\n")

        f.write("# DROPLET CONTOUR COORDINATES (pixel space: x, y)\n")
        for pt in final_polygon:
            f.write(f"{pt[0]:.2f}, {pt[1]:.2f}\n")
        f.write("\n")

        f.write("# PREDICTED CONTOUR COORDINATES (pixel space: x, y)\n")
        for pt in model_px:
            f.write(f"{pt[0]:.2f}, {pt[1]:.2f}\n")

    logger.info(f"Coordinates saved to: {txt_path.resolve()}")

    # --- 7. Verification Assertions ---
    assert abs(diameter_mm - 7.7423) < 0.05, f"Diameter {diameter_mm:.4f} deviated too much"
    assert abs(height_mm - 2.3931) < 0.05, f"Height {height_mm:.4f} deviated too much"
    assert abs(volume_uL - 67.1475) < 0.5, f"Volume {volume_uL:.4f} deviated too much"
    assert abs(contact_angle_deg - 84.2587) < 0.5, f"Contact angle {contact_angle_deg:.4f} deviated too much"
    assert abs(fit_R0_mm - 17.6229) < 0.05, f"Fit R0 {fit_R0_mm:.4f} deviated too much"
    assert abs(fit_beta - 0.0958) < 0.01, f"Fit beta {fit_beta:.4f} deviated too much"
    assert abs(rmse_val - 182.7605) < 0.5, f"RMSE {rmse_val:.4f} deviated too much"
    
    logger.info("Verification checks PASSED successfully. Baseline matches production perfectly!")

if __name__ == "__main__":
    main()
