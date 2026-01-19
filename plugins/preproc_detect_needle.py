"""
Needle detection preprocessor plugin.

This plugin detects the needle region and stores it in the context.
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


def detect_needle_preprocessor(ctx):
    """
    Preprocessor plugin that detects needle region.
    
    For sessile: finds contour touching top border.
    For pendant: uses shaft line analysis.
    
    Stores result in ctx.needle_rect as (x, y, width, height).
    
    Args:
        ctx: Pipeline context with image data
        
    Returns:
        Updated context with needle_rect set.
    """
    if cv2 is None:
        logger.warning("cv2 not available for needle detection")
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
    
    if pipeline == "pendant":
        _detect_needle_pendant(ctx, gray)
    else:
        _detect_needle_sessile(ctx, gray)
    
    return ctx


def _detect_needle_sessile(ctx, gray: np.ndarray) -> None:
    """Detect needle for sessile drop (contour touching top border)."""
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur + adaptive threshold
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 2
    )
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return
    
    # Find contour touching top border
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y < 5:
            ctx.needle_rect = (x, y, w, h)
            logger.info(f"Sessile needle detected: ({x}, {y}, {w}, {h})")
            return


def _detect_needle_pendant(ctx, gray: np.ndarray) -> None:
    """Detect needle for pendant drop using shaft line analysis."""
    height, width = gray.shape[:2]
    
    # First need drop contour
    drop_contour = getattr(ctx, "detected_contour", None)
    if drop_contour is None:
        # Try to detect it
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        
        drop_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(drop_contour)
    pts = drop_contour.reshape(-1, 2)
    
    # Define needle shaft reference (top 20 pixels)
    top_limit = y + 20
    
    # Left shaft line
    left_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] < (x + w/2))]
    if len(left_shaft_pts) == 0:
        return
    ref_x_left = np.median(left_shaft_pts[:, 0])
    
    # Right shaft line
    right_shaft_pts = pts[(pts[:, 1] < top_limit) & (pts[:, 0] > (x + w/2))]
    if len(right_shaft_pts) == 0:
        return
    ref_x_right = np.median(right_shaft_pts[:, 0])
    
    # Create mask for scanning
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [drop_contour], -1, 255, 1)
    
    tolerance = 3
    
    # Find contact points
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
    
    needle_bottom = min(contact_y_left, contact_y_right)
    
    needle_x = int(ref_x_left)
    needle_y = y
    needle_w = int(ref_x_right - ref_x_left)
    needle_h = needle_bottom - y
    
    if needle_w > 0 and needle_h > 0:
        ctx.needle_rect = (needle_x, needle_y, needle_w, needle_h)
        ctx.contact_points = (
            (contact_x_left, contact_y_left),
            (contact_x_right, contact_y_right)
        )
        logger.info(f"Pendant needle detected: ({needle_x}, {needle_y}, {needle_w}, {needle_h})")


# Register as preprocessor plugin
register_preprocessor("detect_needle", detect_needle_preprocessor)
