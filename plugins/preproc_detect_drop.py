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
    """Detect drop for sessile pipeline.
    
    For sessile drops, we require the drop to be in contact with (or very close to)
    the substrate line. This prevents detecting drops hanging from the needle.
    """
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
    
    # Substrate contact tolerance (pixels) - contour bottom must be within this distance
    substrate_touch_tolerance = getattr(ctx, "substrate_touch_tolerance_px", 10)
    
    # Filter valid contours - separate into substrate-touching and non-touching
    substrate_contours = []  # Contours touching substrate (preferred)
    floating_contours = []   # Contours not touching substrate (fallback)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        bottom_y = y + h  # Bottom edge of bounding box
        top_y = y         # Top edge of bounding box
        
        if y < 5:  # Skip needle (touches top of image)
            continue
        
        # ===== NEEDLE FILTER =====
        # For sessile drops, the contour must be COMPLETELY BELOW the needle
        # (not just not overlapping - there should be a clear gap)
        needle_rect = getattr(ctx, "needle_rect", None)
        if needle_rect:
            n_x, n_y, n_w, n_h = needle_rect
            needle_bottom = n_y + n_h
            needle_center_x = n_x + n_w // 2
            
            # STRONG FILTER: Sessile drop must start well BELOW needle bottom
            # If contour top is within 50px of needle bottom, it's likely a pendant drop
            min_gap_from_needle = 50  # pixels
            if top_y < needle_bottom + min_gap_from_needle:
                # This contour is too close to the needle - likely a pendant drop
                logger.debug(f"Skipping contour near needle (top_y={top_y}, needle_bottom={needle_bottom})")
                continue
            
            # Also skip if horizontally centered on needle (pendant drops are aligned with needle)
            cnt_center_x = x + w // 2
            if abs(cnt_center_x - needle_center_x) < n_w and top_y < needle_bottom + 100:
                logger.debug("Skipping contour aligned with needle")
                continue
        
        # ===== ROI RECTANGLE FILTER =====
        # Check both attribute names (roi_rect and roi) for compatibility
        roi_rect = getattr(ctx, "roi_rect", None) or getattr(ctx, "roi", None)
        if roi_rect and len(roi_rect) == 4:
            roi_x, roi_y, roi_w, roi_h = roi_rect
            
            # Skip if contour bounds match ROI bounds (the ROI rectangle itself)
            if (abs(x - roi_x) < 15 and abs(y - roi_y) < 15 and
                abs(w - roi_w) < 15 and abs(h - roi_h) < 15):
                logger.debug("Skipping contour matching ROI rect")
                continue
            
            # Skip contours that extend from ROI top to near substrate (vertical ROI edge)
            if (abs(y - roi_y) < 15 and  # Starts near ROI top
                h > roi_h * 0.7):         # Extends most of ROI height
                logger.debug("Skipping vertical contour (likely ROI edge)")
                continue
        
        # ===== RECTANGULARITY FILTER =====
        # Droplets are curved/elliptical, not rectangular
        rect_area = w * h
        if rect_area > 0:
            rectangularity = area / rect_area
            # Perfect rectangle = 1.0, circle/ellipse ≈ 0.78 (pi/4)
            # Skip if very rectangular (rectangularity > 0.85)
            if rectangularity > 0.85:
                logger.debug(f"Skipping rectangular contour (rect={rectangularity:.2f})")
                continue
        
        # ===== DOME SHAPE VALIDATION =====
        # Sessile drops sit on substrate, not below it
        if substrate_y is not None:
            # The contour should be mostly ABOVE the substrate, not extending far below
            if bottom_y > substrate_y + substrate_touch_tolerance:
                logger.debug("Skipping contour below substrate")
                continue
            # Note: Aspect ratio check removed - ultra-hydrophobic drops can be taller than wide
        
        if area > min_area and x > 5 and (x + w) < (width - 5):
            cnt_center_x = x + w // 2
            distance_from_center = abs(cnt_center_x - center_x)
            
            # Check if contour touches substrate
            if substrate_y is not None:
                distance_to_substrate = abs(bottom_y - substrate_y)
                touches_substrate = distance_to_substrate <= substrate_touch_tolerance
                
                if touches_substrate:
                    substrate_contours.append((cnt, area, distance_from_center, distance_to_substrate))
                else:
                    floating_contours.append((cnt, area, distance_from_center, distance_to_substrate))
            else:
                # No substrate line - use old behavior
                floating_contours.append((cnt, area, distance_from_center, 0))
    
    # Prefer substrate-touching contours
    if substrate_contours:
        # Sort by: closest to substrate, then largest area, then closest to center
        substrate_contours.sort(key=lambda x: (x[3], -x[1], x[2]))
        valid_contours = substrate_contours
        logger.info(f"Found {len(substrate_contours)} substrate-touching contour(s)")
    elif floating_contours:
        # Fallback: use floating contours if no substrate-touching ones
        floating_contours.sort(key=lambda x: (-x[1], x[2]))
        valid_contours = floating_contours
        logger.warning(f"No substrate-touching contours found, using {len(floating_contours)} floating contour(s)")
    else:
        # Last resort fallback
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > min_area:
            hull = cv2.convexHull(largest)
            ctx.detected_contour = hull[:, 0, :].astype(np.float64)
            logger.info("Sessile drop detected (fallback)")
        return
    
    # Select best contour
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
