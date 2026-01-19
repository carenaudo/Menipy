"""
Substrate detection preprocessor plugin.

This plugin detects the substrate baseline and stores it in the context.
Follows the stage-based pattern: operates on ctx and returns ctx.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from menipy.common.registry import register_preprocessor

logger = logging.getLogger(__name__)


def _find_horizon_median(strip_gray: np.ndarray, img_height: int) -> Optional[int]:
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


def detect_substrate_preprocessor(ctx):
    """
    Preprocessor plugin that detects substrate baseline.
    
    Stores result in ctx.substrate_line as ((x1, y1), (x2, y2)).
    
    Args:
        ctx: Pipeline context with image data
        
    Returns:
        Updated context with substrate_line set.
    """
    if cv2 is None:
        logger.warning("cv2 not available for substrate detection")
        return ctx
    
    # Get image from context
    image = getattr(ctx, "image", None)
    if image is None:
        frames = getattr(ctx, "frames", None)
        if frames and len(frames) > 0:
            frame = frames[0]
            image = frame.image if hasattr(frame, "image") else frame
    
    if image is None or not isinstance(image, np.ndarray):
        return ctx
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape[:2]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Calculate margin
    margin_px = max(10, min(50, int(width * 0.05)))
    
    # Extract strips
    left_strip = enhanced[:, 0:margin_px]
    right_strip = enhanced[:, width - margin_px:width]
    
    # Find horizon in each strip
    y_left = _find_horizon_median(left_strip, height)
    y_right = _find_horizon_median(right_strip, height)
    
    if y_left is None and y_right is None:
        substrate_y = int(height * 0.8)
        logger.warning(f"Substrate detection fallback: y={substrate_y}")
    else:
        if y_left is None:
            y_left = y_right
        if y_right is None:
            y_right = y_left
        substrate_y = int((y_left + y_right) / 2)
    
    ctx.substrate_line = ((0, substrate_y), (width, substrate_y))
    logger.info(f"Substrate detected at y={substrate_y}")
    
    return ctx


# Register as preprocessor plugin
register_preprocessor("detect_substrate", detect_substrate_preprocessor)
