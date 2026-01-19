"""
Drop contour detection preprocessor plugin.

This plugin detects the drop contour and stores it in the context.
Follows the stage-based pattern: operates on ctx and returns ctx.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from menipy.common.registry import register_preprocessor

logger = logging.getLogger(__name__)


def detect_drop_preprocessor(ctx):
    """
    Preprocessor plugin that detects drop contour.
    
    For sessile: uses adaptive thresholding, filters by position.
    For pendant: uses Otsu thresholding for high-contrast silhouettes.
    
    Stores result in ctx.detected_contour.
    
    Args:
        ctx: Pipeline context with image data
        
    Returns:
        Updated context with detected_contour set.
    """
    if cv2 is None:
        logger.warning("cv2 not available for drop detection")
        return ctx
    
    # Get image
    image = getattr(ctx, "image", None)
    if image is None:
        frames = getattr(ctx, "frames", None)
        if frames and len(frames) > 0:
            frame = frames[0]
            image = frame.image if hasattr(frame, "image") else frame
    
    if image is None or not isinstance(image, np.ndarray):
        return ctx
    
    # Get pipeline type
    pipeline = getattr(ctx, "pipeline_name", "sessile").lower()
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape[:2]
    image_area = height * width
    
    if pipeline == "pendant":
        _detect_drop_pendant(ctx, gray, height, width, image_area)
    else:
        _detect_drop_sessile(ctx, gray, height, width, image_area)
    
    return ctx


def _detect_drop_sessile(ctx, gray, height, width, image_area) -> None:
    """Detect drop for sessile pipeline."""
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Adaptive threshold
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 2
    )
    
    # Mask below substrate
    substrate_line = getattr(ctx, "substrate_line", None)
    substrate_y = None
    if substrate_line:
        substrate_y = int((substrate_line[0][1] + substrate_line[1][1]) / 2)
        binary[substrate_y - 2:, :] = 0
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return
    
    center_x = width // 2
    min_area = image_area * 0.005
    
    # Filter valid contours
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        if y < 5:  # Skip needle
            continue
        
        if area > min_area and x > 5 and (x + w) < (width - 5):
            cnt_center_x = x + w // 2
            distance_from_center = abs(cnt_center_x - center_x)
            valid_contours.append((cnt, area, distance_from_center))
    
    if not valid_contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > min_area:
            hull = cv2.convexHull(largest)
            ctx.detected_contour = hull[:, 0, :].astype(np.float64)
            logger.info(f"Sessile drop detected (fallback)")
        return
    
    # Select best
    valid_contours.sort(key=lambda x: (-x[1], x[2]))
    best_cnt = valid_contours[0][0]
    
    # Apply convex hull
    hull = cv2.convexHull(best_cnt)
    points = hull[:, 0, :]
    
    # Store contour
    ctx.detected_contour = points.astype(np.float64)
    
    # Calculate contact points if substrate known
    if substrate_y is not None:
        dome_points = [pt for pt in points if pt[1] < (substrate_y - 5)]
        if dome_points:
            dome_points = sorted(dome_points, key=lambda p: p[0])
            ctx.contact_points = (
                (int(dome_points[0][0]), substrate_y),
                (int(dome_points[-1][0]), substrate_y)
            )
    
    logger.info(f"Sessile drop detected with {len(points)} points")


def _detect_drop_pendant(ctx, gray, height, width, image_area) -> None:
    """Detect drop for pendant pipeline."""
    # Otsu threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return
    
    img_center_x = width // 2
    min_area = image_area * 0.05
    
    # Filter valid contours
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            if abs(cx - img_center_x) < (width * 0.3):
                valid_contours.append((cnt, area))
    
    if not valid_contours:
        return
    
    # Select largest
    drop_cnt = max(valid_contours, key=lambda x: x[1])[0]
    ctx.detected_contour = drop_cnt
    
    # Detect apex (bottom point)
    pts = drop_cnt.reshape(-1, 2)
    apex_idx = np.argmax(pts[:, 1])
    ctx.apex_point = (int(pts[apex_idx][0]), int(pts[apex_idx][1]))
    
    logger.info(f"Pendant drop detected with {len(drop_cnt)} points")


# Register as preprocessor plugin
register_preprocessor("detect_drop", detect_drop_preprocessor)
