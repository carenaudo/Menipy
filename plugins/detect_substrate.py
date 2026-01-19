"""
Substrate detection plugin using auto-calibration algorithms.

Provides automatic substrate/baseline detection for sessile drop analysis.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np

from menipy.common.registry import register_substrate_detector

logger = logging.getLogger(__name__)


def detect_substrate_gradient(
    image: np.ndarray,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: Tuple[int, int] = (8, 8),
    margin_fraction: float = 0.05,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Detect substrate baseline using gradient analysis on image margins.
    
    Finds the strongest negative gradient (dark-to-light transition) in the
    left and right margins of the image.
    
    Args:
        image: Input image (BGR or grayscale)
        clahe_clip_limit: CLAHE clip limit for contrast enhancement
        clahe_tile_size: CLAHE tile grid size
        margin_fraction: Fraction of image width for margin analysis
        
    Returns:
        Substrate line as ((x1, y1), (x2, y2)) or None if not found.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape[:2]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
    enhanced = clahe.apply(gray)
    
    # Calculate margin size
    margin_px = max(10, min(50, int(width * margin_fraction)))
    
    # Extract left and right strips
    left_strip = enhanced[:, 0:margin_px]
    right_strip = enhanced[:, width - margin_px:width]
    
    # Find horizon in each strip
    y_left = _find_horizon_median(left_strip, height)
    y_right = _find_horizon_median(right_strip, height)
    
    if y_left is None and y_right is None:
        # Fallback: assume substrate at bottom 20%
        substrate_y = int(height * 0.8)
        logger.warning(f"Substrate detection fallback: y={substrate_y}")
        return ((0, substrate_y), (width, substrate_y))
    
    if y_left is None:
        y_left = y_right
    if y_right is None:
        y_right = y_left
    
    substrate_y = int((y_left + y_right) / 2)
    
    logger.info(f"Substrate detected at y={substrate_y} (left={y_left}, right={y_right})")
    return ((0, substrate_y), (width, substrate_y))


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
        
        # Find strongest negative gradient (dark-to-light transition)
        best_idx = np.argmin(valid_grad)
        best_y = best_idx + min_limit
        detected_ys.append(best_y)
    
    if not detected_ys:
        return None
    
    return int(np.median(detected_ys))


def detect_substrate_hough(
    image: np.ndarray,
    *,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 100,
    angle_tolerance: float = 5.0,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Detect substrate baseline using Hough line transform.
    
    Finds horizontal lines near the bottom of the image.
    
    Args:
        image: Input image (BGR or grayscale)
        canny_low: Canny edge detection low threshold
        canny_high: Canny edge detection high threshold
        hough_threshold: Hough accumulator threshold
        angle_tolerance: Angle tolerance from horizontal (degrees)
        
    Returns:
        Substrate line as ((x1, y1), (x2, y2)) or None if not found.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape[:2]
    
    # Focus on bottom half of image
    bottom_half = gray[height // 2:, :]
    
    # Edge detection
    edges = cv2.Canny(bottom_half, canny_low, canny_high)
    
    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_threshold)
    
    if lines is None:
        return None
    
    # Filter for horizontal lines
    horizontal_lines = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta)
        
        # Horizontal lines have theta near 0 or 180
        if abs(angle_deg - 90) < angle_tolerance:
            y = int(rho)
            # Adjust for bottom half offset
            actual_y = y + height // 2
            if height * 0.5 < actual_y < height * 0.95:
                horizontal_lines.append(actual_y)
    
    if not horizontal_lines:
        return None
    
    # Take median of detected lines
    substrate_y = int(np.median(horizontal_lines))
    
    logger.info(f"Substrate detected via Hough at y={substrate_y}")
    return ((0, substrate_y), (width, substrate_y))


# Register plugins
register_substrate_detector("gradient", detect_substrate_gradient)
register_substrate_detector("hough", detect_substrate_hough)
