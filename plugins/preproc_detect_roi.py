"""
ROI detection preprocessor plugin.

This plugin detects the region of interest and stores it in the context.
Follows the stage-based pattern: operates on ctx and returns ctx.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from menipy.common.registry import register_preprocessor

logger = logging.getLogger(__name__)


def detect_roi_preprocessor(ctx):
    """
    Preprocessor plugin that detects ROI from other detected features.
    
    Uses detected drop, substrate, needle to compute bounding box.
    Should run AFTER other detection preprocessors.
    
    Stores result in ctx.detected_roi as (x, y, width, height).
    
    Args:
        ctx: Pipeline context with detected features
        
    Returns:
        Updated context with detected_roi set.
    """
    # Get image dimensions
    image = getattr(ctx, "image", None)
    if image is None:
        frames = getattr(ctx, "frames", None)
        if frames and len(frames) > 0:
            frame = frames[0]
            image = frame.image if hasattr(frame, "image") else frame
    
    if image is None or not isinstance(image, np.ndarray):
        return ctx
    
    if len(image.shape) == 3:
        height, width = image.shape[:2]
    else:
        height, width = image.shape
    
    # Get pipeline type
    pipeline = getattr(ctx, "pipeline_name", "sessile").lower()
    
    if pipeline == "pendant":
        _compute_roi_pendant(ctx, height, width)
    else:
        _compute_roi_sessile(ctx, height, width)
    
    return ctx


def _compute_roi_sessile(ctx, height: int, width: int, padding: int = 20) -> None:
    """Compute ROI for sessile pipeline."""
    x_min, y_min = width, height
    x_max, y_max = 0, 0
    has_data = False
    
    # Include drop contour
    drop_contour = getattr(ctx, "detected_contour", None)
    if drop_contour is not None and len(drop_contour) > 0:
        contour = np.asarray(drop_contour)
        if contour.ndim == 3:
            contour = contour.reshape(-1, 2)
        x_min = min(x_min, int(np.min(contour[:, 0])))
        x_max = max(x_max, int(np.max(contour[:, 0])))
        y_min = min(y_min, int(np.min(contour[:, 1])))
        y_max = max(y_max, int(np.max(contour[:, 1])))
        has_data = True
    
    # Include substrate
    substrate_line = getattr(ctx, "substrate_line", None)
    if substrate_line:
        substrate_y = int((substrate_line[0][1] + substrate_line[1][1]) / 2)
        y_max = max(y_max, substrate_y)
        has_data = True
    
    # Include needle (bottom 30%)
    needle_rect = getattr(ctx, "needle_rect", None)
    if needle_rect:
        nx, ny, nw, nh = needle_rect
        needle_include_y = ny + int(nh * 0.7)
        y_min = min(y_min, needle_include_y)
        x_min = min(x_min, nx)
        x_max = max(x_max, nx + nw)
        has_data = True
    
    if not has_data:
        return
    
    # Apply padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    
    if roi_width > 0 and roi_height > 0:
        ctx.detected_roi = (x_min, y_min, roi_width, roi_height)
        logger.info(f"Sessile ROI: ({x_min}, {y_min}, {roi_width}, {roi_height})")


def _compute_roi_pendant(ctx, height: int, width: int, padding: int = 20) -> None:
    """Compute ROI for pendant pipeline."""
    drop_contour = getattr(ctx, "detected_contour", None)
    if drop_contour is None or len(drop_contour) == 0:
        return
    
    contour = np.asarray(drop_contour)
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)
    
    x_min = int(np.min(contour[:, 0]))
    x_max = int(np.max(contour[:, 0]))
    y_min = int(np.min(contour[:, 1]))
    y_max = int(np.max(contour[:, 1]))
    
    # Include apex
    apex_point = getattr(ctx, "apex_point", None)
    if apex_point:
        y_max = max(y_max, apex_point[1])
    
    # Apply padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    
    if roi_width > 0 and roi_height > 0:
        ctx.detected_roi = (x_min, y_min, roi_width, roi_height)
        logger.info(f"Pendant ROI: ({x_min}, {y_min}, {roi_width}, {roi_height})")


# Register as preprocessor plugin
register_preprocessor("detect_roi", detect_roi_preprocessor)
