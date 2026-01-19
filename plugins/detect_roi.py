"""
ROI detection plugin using auto-calibration algorithms.

Provides automatic Region of Interest detection for droplet analysis.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from menipy.common.registry import register_roi_detector

logger = logging.getLogger(__name__)


def detect_roi_sessile(
    image: np.ndarray,
    *,
    drop_contour: Optional[np.ndarray] = None,
    substrate_y: Optional[int] = None,
    needle_rect: Optional[Tuple[int, int, int, int]] = None,
    padding: int = 20,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect ROI for sessile drop (bounding box around drop + substrate).
    
    Args:
        image: Input image (BGR or grayscale)
        drop_contour: Pre-detected drop contour
        substrate_y: Y-coordinate of substrate line
        needle_rect: Needle bounding box (x, y, w, h)
        padding: Padding pixels around detected regions
        
    Returns:
        ROI as (x, y, width, height) or None if insufficient data.
    """
    if len(image.shape) == 3:
        height, width = image.shape[:2]
    else:
        height, width = image.shape
    
    x_min, y_min = width, height
    x_max, y_max = 0, 0
    has_data = False
    
    # Include drop contour
    if drop_contour is not None and len(drop_contour) > 0:
        contour = np.asarray(drop_contour)
        if contour.ndim == 3:
            contour = contour.reshape(-1, 2)
        x_min = min(x_min, int(np.min(contour[:, 0])))
        x_max = max(x_max, int(np.max(contour[:, 0])))
        y_min = min(y_min, int(np.min(contour[:, 1])))
        y_max = max(y_max, int(np.max(contour[:, 1])))
        has_data = True
    
    # Include substrate line
    if substrate_y is not None:
        y_max = max(y_max, substrate_y)
        has_data = True
    
    # Include needle (bottom 30%)
    if needle_rect is not None:
        nx, ny, nw, nh = needle_rect
        needle_include_y = ny + int(nh * 0.7)
        y_min = min(y_min, needle_include_y)
        x_min = min(x_min, nx)
        x_max = max(x_max, nx + nw)
        has_data = True
    
    if not has_data:
        logger.warning("ROI detection: no data available")
        return None
    
    # Apply padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    
    if roi_width <= 0 or roi_height <= 0:
        return None
    
    logger.info(f"Sessile ROI detected: ({x_min}, {y_min}, {roi_width}, {roi_height})")
    return (x_min, y_min, roi_width, roi_height)


def detect_roi_pendant(
    image: np.ndarray,
    *,
    drop_contour: Optional[np.ndarray] = None,
    apex_point: Optional[Tuple[int, int]] = None,
    padding: int = 20,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect ROI for pendant drop (from needle to apex).
    
    Args:
        image: Input image (BGR or grayscale)
        drop_contour: Pre-detected drop contour
        apex_point: Apex point (x, y)
        padding: Padding pixels around detected regions
        
    Returns:
        ROI as (x, y, width, height) or None if insufficient data.
    """
    if drop_contour is None or len(drop_contour) == 0:
        logger.warning("Pendant ROI detection requires drop contour")
        return None
    
    if len(image.shape) == 3:
        height, width = image.shape[:2]
    else:
        height, width = image.shape
    
    contour = np.asarray(drop_contour)
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)
    
    x_min = int(np.min(contour[:, 0]))
    x_max = int(np.max(contour[:, 0]))
    y_min = int(np.min(contour[:, 1]))
    y_max = int(np.max(contour[:, 1]))
    
    # Include apex
    if apex_point is not None:
        y_max = max(y_max, apex_point[1])
    
    # Apply padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min)  # Start from top of drop
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    
    if roi_width <= 0 or roi_height <= 0:
        return None
    
    logger.info(f"Pendant ROI detected: ({x_min}, {y_min}, {roi_width}, {roi_height})")
    return (x_min, y_min, roi_width, roi_height)


def detect_roi_auto(
    image: np.ndarray,
    pipeline: str = "sessile",
    **kwargs
) -> Optional[Tuple[int, int, int, int]]:
    """
    Auto-detect ROI based on pipeline type.
    
    Args:
        image: Input image
        pipeline: Pipeline name ("sessile" or "pendant")
        **kwargs: Additional parameters passed to specific detector
        
    Returns:
        ROI as (x, y, width, height) or None.
    """
    if pipeline.lower() == "pendant":
        return detect_roi_pendant(image, **kwargs)
    else:
        return detect_roi_sessile(image, **kwargs)


# Register plugins
register_roi_detector("sessile", detect_roi_sessile)
register_roi_detector("pendant", detect_roi_pendant)
register_roi_detector("auto", detect_roi_auto)
