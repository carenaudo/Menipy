"""
Contour refinement module for droplet shape analysis.

This module coordinates sub-pixel contour refinement between coarse edge extraction
and downstream physics/geometric fitting. It supports:
1. Active contour (Snake) optimization against raw image gradients with substrate
   sliding line boundary constraints (Kass et al., 1988; Stalder et al., 2010).
2. Parametric cubic B-spline curve fitting with analytical derivatives and contact
   angles (Brigger et al., 2000; Unser, 1993).
3. Legacy Savitzky-Golay polynomial smoothing (Savitzky & Golay, 1964).
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from menipy.common.contour_smoothing import smooth_contour
from menipy.common.geometry import estimate_contact_angle_tangent
from menipy.math.active_contour import (
    ActiveContourConfig,
    SnakeBoundaryCondition,
    evolve_active_contour,
    fit_bspline_snake,
)
from menipy.models.config import ContourSmoothingSettings
from menipy.models.context import Context
from menipy.models.geometry import Contour

logger = logging.getLogger(__name__)


def _extract_image_frame(ctx: Context) -> np.ndarray | None:
    """Extract a 2D grayscale image array from Context.

    Checks current_frame, frames, image, preprocessed, and gray fields.

    Args:
        ctx: Pipeline context.

    Returns:
        2D uint8 or float grayscale image array, or None if unavailable.
    """
    img = None

    # Check current_frame
    current_frame = getattr(ctx, "current_frame", None)
    if current_frame is not None:
        if hasattr(current_frame, "image") and current_frame.image is not None:
            img = current_frame.image
        elif isinstance(current_frame, np.ndarray):
            img = current_frame

    # Check frames
    if img is None:
        frames = getattr(ctx, "frames", None)
        if frames is not None:
            if isinstance(frames, np.ndarray):
                if frames.ndim >= 2:
                    img = frames[0] if frames.ndim > 2 and len(frames) > 0 else frames
            elif isinstance(frames, (list, tuple)) and len(frames) > 0:
                first = frames[0]
                if hasattr(first, "image") and first.image is not None:
                    img = first.image
                elif isinstance(first, np.ndarray):
                    img = first

    # Check image attribute
    if img is None:
        img_attr = getattr(ctx, "image", None)
        if isinstance(img_attr, np.ndarray):
            img = img_attr

    # Check preprocessed
    if img is None:
        preprocessed = getattr(ctx, "preprocessed", None)
        if isinstance(preprocessed, np.ndarray):
            img = preprocessed

    # Check gray
    if img is None:
        gray = getattr(ctx, "gray", None)
        if isinstance(gray, np.ndarray):
            img = gray

    if img is None:
        return None

    # Convert to 2D grayscale if 3D
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 1:
            img = img.squeeze(axis=2)

    return img


def _compute_substrate_line_eq(
    ctx: Context,
    contour_xy: np.ndarray,
) -> tuple[float, float, float]:
    """Compute normalized line equation coefficients (A, B, C) where A*x + B*y + C = 0.

    Args:
        ctx: Pipeline context.
        contour_xy: (N, 2) array of contour coordinates.

    Returns:
        (A, B, C) tuple with A^2 + B^2 = 1.
    """
    substrate_line = getattr(ctx, "substrate_line", None)
    if substrate_line is not None:
        p1, p2 = substrate_line
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length > 1e-6:
            # Line direction is (dx, dy), normal is (-dy, dx)
            A = -dy / length
            B = dx / length
            C = -(A * x1 + B * y1)
            return (A, B, C)

    # Check geometry.baseline_y
    geometry = getattr(ctx, "geometry", None)
    if geometry and getattr(geometry, "baseline_y", None) is not None:
        baseline_y = float(geometry.baseline_y)
        return (0.0, 1.0, -baseline_y)

    # Fallback: horizontal line through maximum y
    max_y = float(np.max(contour_xy[:, 1]))
    return (0.0, 1.0, -max_y)


def _compute_contact_angles(
    pts: np.ndarray,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None,
    bspline_angles: tuple[float, float] | None,
) -> tuple[float | None, float | None]:
    """Compute robust left and right contact angles from refined points.

    Uses flank polynomial tangent fitting when substrate line is available to avoid
    2D corner blunting artifacts at the solid substrate (Stalder et al., 2010),
    falling back to analytical B-spline endpoint tangents.

    Args:
        pts: (M, 2) array of refined contour points ordered left-to-right.
        substrate_line: Substrate line segment ((x1, y1), (x2, y2)) or None.
        bspline_angles: (theta_left, theta_right) from analytical B-spline or None.

    Returns:
        (left_angle_deg, right_angle_deg)
    """
    left_angle = bspline_angles[0] if bspline_angles is not None else None
    right_angle = bspline_angles[1] if bspline_angles is not None else None

    if substrate_line is not None and len(pts) >= 10:
        try:
            ang_l, _ = estimate_contact_angle_tangent(pts, pts[0], substrate_line=substrate_line)
            ang_r, _ = estimate_contact_angle_tangent(pts, pts[-1], substrate_line=substrate_line)
            if ang_l is not None and not np.isnan(ang_l):
                left_angle = float(ang_l)
            if ang_r is not None and not np.isnan(ang_r):
                right_angle = float(ang_r)
        except Exception as e:
            logger.debug("Flank tangent angle estimation failed: %s; using B-spline angles", e)

    return left_angle, right_angle


def refine_contour(
    ctx: Context,
    settings: ContourSmoothingSettings | None = None,
) -> Context:
    """Refine droplet contour and contact points using configured refinement method.

    Supports:
    - active_contour: Sub-pixel active contour energy minimization on raw image gradients
      with substrate sliding line constraints, followed by B-spline tangent evaluation.
    - bspline: Continuous parametric cubic B-spline curve fitting with analytical derivatives.
    - savgol: Legacy Savitzky-Golay moving-window polynomial filter.

    Updates:
    - ctx.contour
    - ctx.sessile_calc_contour
    - ctx.sessile_calc_contact_points
    - ctx.contact_points
    - ctx.smoothing_results

    Args:
        ctx: Pipeline context containing extracted contour and substrate line.
        settings: Optional ContourSmoothingSettings. If None, reads from ctx or uses defaults.

    Returns:
        Updated pipeline context.
    """
    if settings is None:
        settings = getattr(ctx, "contour_smoothing_settings", None)
    if settings is None:
        settings = ContourSmoothingSettings()

    if not settings.enabled:
        return ctx

    # Validate contour
    contour = getattr(ctx, "contour", None)
    if contour is None or contour.xy is None:
        logger.warning("No contour available in context for refinement")
        return ctx

    contour_xy = np.asarray(contour.xy, dtype=float)
    if contour_xy.ndim != 2 or contour_xy.shape[0] < 4:
        logger.warning(
            "Insufficient contour points for refinement: shape=%s", contour_xy.shape
        )
        return ctx

    line_eq = _compute_substrate_line_eq(ctx, contour_xy)
    substrate_line = getattr(ctx, "substrate_line", None)
    refined_xy: np.ndarray | None = None
    smoothing_dict: dict[str, Any] = {"method": settings.method}

    if settings.method == "active_contour":
        image = _extract_image_frame(ctx)
        if image is not None:
            try:
                snake_cfg = ActiveContourConfig(
                    alpha=settings.snake_alpha,
                    beta=settings.snake_beta,
                    gamma=settings.snake_gamma,
                    w_edge=settings.snake_w_edge,
                    w_line=settings.snake_w_line,
                    w_balloon=settings.snake_w_balloon,
                    max_iterations=settings.snake_max_iterations,
                    convergence=settings.snake_convergence,
                )

                snake_res = evolve_active_contour(
                    image=image,
                    init_xy=contour_xy,
                    config=snake_cfg,
                    boundary_condition=SnakeBoundaryCondition.SLIDING_LINE,
                    substrate_line=line_eq,
                )

                bspline_res = fit_bspline_snake(
                    xy=snake_res.xy,
                    substrate_line=line_eq,
                    num_eval_points=settings.spline_eval_points,
                    smoothing=settings.spline_smoothing,
                )

                refined_xy = bspline_res.xy
                left_angle, right_angle = _compute_contact_angles(
                    refined_xy, substrate_line, bspline_res.contact_angles_deg
                )

                smoothing_dict.update(
                    {
                        "converged": snake_res.converged,
                        "iterations": snake_res.iterations,
                        "energy": snake_res.energy,
                        "left_contact": refined_xy[0],
                        "right_contact": refined_xy[-1],
                        "left_angle_deg": left_angle,
                        "right_angle_deg": right_angle,
                        "xy": refined_xy,
                    }
                )
                logger.info(
                    "Active contour refinement succeeded: %d iterations, energy=%.4f, angles L=%s R=%s",
                    snake_res.iterations,
                    snake_res.energy,
                    f"{left_angle:.1f}°" if left_angle is not None else "N/A",
                    f"{right_angle:.1f}°" if right_angle is not None else "N/A",
                )
            except Exception as e:
                logger.warning(
                    "Active contour refinement failed: %s; falling back to B-spline", e
                )
                refined_xy = None

        if refined_xy is None:
            # Fallback to B-spline if no image or snake error
            logger.info("Running B-spline refinement without image gradients")
            try:
                bspline_res = fit_bspline_snake(
                    xy=contour_xy,
                    substrate_line=line_eq,
                    num_eval_points=settings.spline_eval_points,
                    smoothing=settings.spline_smoothing,
                )
                refined_xy = bspline_res.xy
                left_angle, right_angle = _compute_contact_angles(
                    refined_xy, substrate_line, bspline_res.contact_angles_deg
                )

                smoothing_dict.update(
                    {
                        "fallback": "bspline",
                        "left_contact": refined_xy[0],
                        "right_contact": refined_xy[-1],
                        "left_angle_deg": left_angle,
                        "right_angle_deg": right_angle,
                        "xy": refined_xy,
                    }
                )
            except Exception as e:
                logger.warning("B-spline refinement failed: %s", e)

    elif settings.method == "bspline":
        try:
            bspline_res = fit_bspline_snake(
                xy=contour_xy,
                substrate_line=line_eq,
                num_eval_points=settings.spline_eval_points,
                smoothing=settings.spline_smoothing,
            )
            refined_xy = bspline_res.xy
            left_angle = (
                bspline_res.contact_angles_deg[0]
                if bspline_res.contact_angles_deg
                else None
            )
            right_angle = (
                bspline_res.contact_angles_deg[1]
                if bspline_res.contact_angles_deg
                else None
            )

            smoothing_dict.update(
                {
                    "left_contact": refined_xy[0],
                    "right_contact": refined_xy[-1],
                    "left_angle_deg": left_angle,
                    "right_angle_deg": right_angle,
                    "xy": refined_xy,
                }
            )
            logger.info(
                "B-spline refinement succeeded: %d points, angles L=%s R=%s",
                len(refined_xy),
                f"{left_angle:.1f}°" if left_angle is not None else "N/A",
                f"{right_angle:.1f}°" if right_angle is not None else "N/A",
            )
        except Exception as e:
            logger.warning("B-spline refinement failed: %s", e)

    elif settings.method == "savgol":
        # Determine baseline Y for savgol
        if substrate_line is not None:
            p1, p2 = substrate_line
            substrate_y = float((p1[1] + p2[1]) / 2)
        else:
            substrate_y = float(np.max(contour_xy[:, 1]))

        savgol_res = smooth_contour(
            contour_xy,
            substrate_y,
            window_length=settings.window_length,
            polyorder=settings.polyorder,
            filter_monotonic=settings.filter_monotonic,
            filter_below_substrate=settings.filter_below_substrate,
            extrapolate_contact_points=settings.extrapolate_contact_points,
        )
        if savgol_res is not None:
            smoothing_dict.update(savgol_res)
            if "x_smooth" in savgol_res and "y_smooth" in savgol_res:
                refined_xy = np.column_stack(
                    (savgol_res["x_smooth"], savgol_res["y_smooth"])
                )

    if refined_xy is not None and len(refined_xy) >= 2:
        ctx.contour = Contour(xy=refined_xy)
        ctx.sessile_calc_contour = np.asarray(refined_xy, dtype=float)
        p_left = refined_xy[0]
        p_right = refined_xy[-1]
        ctx.sessile_calc_contact_points = (
            (float(p_left[0]), float(p_left[1])),
            (float(p_right[0]), float(p_right[1])),
        )
        ctx.contact_points = (
            (int(round(p_left[0])), int(round(p_left[1]))),
            (int(round(p_right[0])), int(round(p_right[1]))),
        )
        ctx.smoothing_results = smoothing_dict
    else:
        logger.warning(
            "Contour refinement could not produce refined points; preserving original contour"
        )

    return ctx


# Alias run for pipeline stage compatibility
run = refine_contour
