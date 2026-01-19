"""
Needle detection plugin using auto-calibration algorithms.

Provides needle detection for both sessile and pendant pipelines.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from menipy.common.registry import register_needle_detector

logger = logging.getLogger(__name__)


def detect_needle_sessile(
    image: np.ndarray,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: Tuple[int, int] = (8, 8),
    adaptive_block_size: int = 21,
    adaptive_c: int = 2,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect needle region for sessile drop (contour touching top border).
    
    Args:
        image: Input image (BGR or grayscale)
        clahe_clip_limit: CLAHE clip limit for contrast enhancement
        clahe_tile_size: CLAHE tile grid size
        adaptive_block_size: Block size for adaptive thresholding
        adaptive_c: Constant for adaptive thresholding
        
    Returns:
        Needle bounding box as (x, y, width, height) or None if not found.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
    enhanced = clahe.apply(gray)
    
    # Gaussian blur + adaptive threshold
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c
    )
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find contour touching top border
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y < 5:  # Touches top border
            logger.info(f"Sessile needle detected: ({x}, {y}, {w}, {h})")
            return (x, y, w, h)
    
    return None


def detect_needle_pendant(
    image: np.ndarray,
    drop_contour: Optional[np.ndarray] = None,
    *,
    tolerance: int = 3,
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Detect needle region for pendant drop using shaft line analysis.
    
    Args:
        image: Input image (BGR or grayscale)
        drop_contour: Pre-detected drop contour (required)
        tolerance: Pixels deviation to detect contact points
        
    Returns:
        Tuple of (needle_rect, contact_points) or None if not found.
        needle_rect is (x, y, width, height)
        contact_points is ((left_x, left_y), (right_x, right_y))
    """
    if drop_contour is None:
        logger.warning("Pendant needle detection requires drop contour")
        return None
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape[:2]
    
    x, y, w, h = cv2.boundingRect(drop_contour)
    pts = drop_contour.reshape(-1, 2)
    
    # Define needle shaft reference (top 20 pixels)
    top_limit = y + 20
    
    # Left shaft line
    left_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] < (x + w/2))]
    if len(left_shaft_pts) == 0:
        return None
    ref_x_left = np.median(left_shaft_pts[:, 0])
    
    # Right shaft line
    right_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] > (x + w/2))]
    if len(right_shaft_pts) == 0:
        return None
    ref_x_right = np.median(right_shaft_pts[:, 0])
    
    # Create mask for scanning
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [drop_contour], -1, 255, 1)
    
    # Find left contact point
    contact_y_left = y
    contact_x_left = int(ref_x_left)
    
    for cy in range(y, y + h):
        row = mask[cy, 0:int(x + w/2)]
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            current_x = indices[0]
            if current_x < (ref_x_left - tolerance):
                contact_y_left = cy
                contact_x_left = current_x
                break
    
    # Find right contact point
    contact_y_right = y
    contact_x_right = int(ref_x_right)
    
    for cy in range(y, y + h):
        row = mask[cy, int(x + w/2):width]
        indices = np.where(row > 0)[0]
        if len(indices) > 0:
            current_x = indices[-1] + int(x + w/2)
            if current_x > (ref_x_right + tolerance):
                contact_y_right = cy
                contact_x_right = current_x
                break
    
    # Needle bottom
    needle_bottom = min(contact_y_left, contact_y_right)
    
    needle_x = int(ref_x_left)
    needle_y = y
    needle_w = int(ref_x_right - ref_x_left)
    needle_h = needle_bottom - y
    
    if needle_w <= 0 or needle_h <= 0:
        return None
    
    needle_rect = (needle_x, needle_y, needle_w, needle_h)
    contact_points = (
        (contact_x_left, contact_y_left),
        (contact_x_right, contact_y_right)
    )
    
    logger.info(f"Pendant needle detected: {needle_rect}")
    return (needle_rect, contact_points)


# Register plugins
register_needle_detector("sessile", detect_needle_sessile)
register_needle_detector("pendant", detect_needle_pendant)
