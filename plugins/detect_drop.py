"""
Drop contour detection plugin using auto-calibration algorithms.

Provides automatic drop contour detection for droplet analysis.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from menipy.common.registry import register_drop_detector

logger = logging.getLogger(__name__)


def detect_drop_sessile(
    image: np.ndarray,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: Tuple[int, int] = (8, 8),
    adaptive_block_size: int = 21,
    adaptive_c: int = 2,
    substrate_y: Optional[int] = None,
    min_area_fraction: float = 0.005,
) -> Optional[Tuple[np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Detect drop contour for sessile drop using adaptive thresholding.
    
    Args:
        image: Input image (BGR or grayscale)
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_size: CLAHE tile grid size
        adaptive_block_size: Block size for adaptive thresholding
        adaptive_c: Constant for adaptive thresholding
        substrate_y: Y-coordinate of substrate line
        min_area_fraction: Minimum contour area as fraction of image
        
    Returns:
        Tuple of (drop_contour, contact_points) or None.
        drop_contour is Nx2 array, contact_points is ((left), (right))
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape[:2]
    image_area = height * width
    
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
    
    # Mask below substrate line
    if substrate_y is not None:
        binary[substrate_y - 2:, :] = 0
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    center_x = width // 2
    min_area = image_area * min_area_fraction
    
    # Filter for valid drop contours
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Skip needle (touches top)
        if y < 5:
            continue
        
        # Filter by area and position
        if area > min_area and x > 5 and (x + w) < (width - 5):
            cnt_center_x = x + w // 2
            distance_from_center = abs(cnt_center_x - center_x)
            valid_contours.append((cnt, area, distance_from_center))
    
    if not valid_contours:
        # Fallback to largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > min_area:
            hull = cv2.convexHull(largest)
            return hull[:, 0, :].astype(np.float64), None
        return None
    
    # Select best contour
    valid_contours.sort(key=lambda x: (-x[1], x[2]))
    best_cnt = valid_contours[0][0]
    
    # Apply convex hull
    hull = cv2.convexHull(best_cnt)
    points = hull[:, 0, :]
    
    # Reconstruct with flat base at substrate
    if substrate_y is not None:
        dome_points = [pt for pt in points if pt[1] < (substrate_y - 5)]
        
        if dome_points:
            dome_points = sorted(dome_points, key=lambda p: p[0])
            x_left = dome_points[0][0]
            x_right = dome_points[-1][0]
            
            cp_left = (int(x_left), substrate_y)
            cp_right = (int(x_right), substrate_y)
            contact_points = (cp_left, cp_right)
            
            final_polygon = np.array(
                [[x_left, substrate_y]] + 
                [[p[0], p[1]] for p in dome_points] + 
                [[x_right, substrate_y]],
                dtype=np.float64
            )
            
            logger.info(f"Sessile drop detected with {len(final_polygon)} points")
            return final_polygon, contact_points
    
    logger.info(f"Sessile drop detected with {len(points)} points (no substrate)")
    return points.astype(np.float64), None


def detect_drop_pendant(
    image: np.ndarray,
    *,
    min_area_fraction: float = 0.05,
) -> Optional[np.ndarray]:
    """
    Detect drop contour for pendant drop using Otsu thresholding.
    
    Args:
        image: Input image (BGR or grayscale)
        min_area_fraction: Minimum contour area as fraction of image
        
    Returns:
        Drop contour as OpenCV contour array or None.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape[:2]
    image_area = height * width
    
    # Gaussian blur + Otsu threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    img_center_x = width // 2
    min_area = image_area * min_area_fraction
    
    # Filter for valid contours
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            # Must be roughly centered
            if abs(cx - img_center_x) < (width * 0.3):
                valid_contours.append((cnt, area))
    
    if not valid_contours:
        return None
    
    # Select largest valid contour
    drop_cnt = max(valid_contours, key=lambda x: x[1])[0]
    
    logger.info(f"Pendant drop detected with {len(drop_cnt)} points")
    return drop_cnt


# Register plugins
register_drop_detector("sessile", detect_drop_sessile)
register_drop_detector("pendant", detect_drop_pendant)
