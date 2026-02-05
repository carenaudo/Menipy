"""
Drop contour detection plugin using auto-calibration algorithms.

Provides automatic drop contour detection for droplet analysis.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
# NOTE: cv2 import moved inside functions

from menipy.common.registry import register_drop_detector
from pydantic import BaseModel, Field, ConfigDict
from menipy.common.plugin_settings import register_detector_settings, resolve_plugin_settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration Models
# -----------------------------------------------------------------------------

class DropSessileSettings(BaseModel):
    """Settings for sessile drop detection."""
    model_config = ConfigDict(extra='ignore')
    
    clahe_clip_limit: float = Field(2.0, description="Contrast enhancement limit")
    adaptive_block_size: int = Field(21, description="Threshold block size (odd)")
    adaptive_c: int = Field(2, description="Threshold constant")
    min_area_fraction: float = Field(0.005, description="Min area as fraction of image")
    substrate_touch_tolerance: int = Field(10, description="Max pixels from substrate to be considered touching")
    rectangularity_threshold: float = Field(0.85, description="Max rectangularity to filter out ROI boxes")

    def model_post_init(self, __context):
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1

class DropPendantSettings(BaseModel):
    """Settings for pendant drop detection."""
    model_config = ConfigDict(extra='ignore')
    
    min_area_fraction: float = Field(0.05, description="Min area as fraction of image")
    centering_tolerance: float = Field(0.3, description="Max horizontal offset from center (fraction of width)")


# -----------------------------------------------------------------------------
# Implementations
# -----------------------------------------------------------------------------

def detect_drop_sessile(
    image: np.ndarray,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: Tuple[int, int] = (8, 8),
    adaptive_block_size: int = 21,
    adaptive_c: int = 2,
    substrate_y: Optional[int] = None,
    min_area_fraction: float = 0.005,
    **kwargs
) -> Optional[Tuple[np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Detect drop contour for sessile drop using adaptive thresholding.
    """
    import cv2 

    raw_cfg = resolve_plugin_settings("sessile", kwargs.get("plugin_settings", {}), **kwargs)
    cfg = DropSessileSettings(**raw_cfg)

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape[:2]
    image_area = height * width
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=clahe_tile_size)
    enhanced = clahe.apply(gray)
    
    # Gaussian blur + adaptive threshold
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg.adaptive_block_size,
        cfg.adaptive_c
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
    min_area = image_area * cfg.min_area_fraction
    
    # Filter valid contours - separate into substrate-touching and non-touching
    substrate_contours = []  # Contours touching substrate (preferred)
    floating_contours = []   # Contours not touching substrate (fallback)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        bottom_y = y + h  # Bottom edge of bounding box
        
        # Skip needle (touches top)
        if y < 5:
            continue
        
        # Skip rectangular contours (likely ROI boundaries, not droplets)
        rect_area = w * h
        if rect_area > 0:
            rectangularity = area / rect_area
            # Perfect rectangle = 1.0, circle/ellipse ≈ 0.78 (pi/4)
            if rectangularity > cfg.rectangularity_threshold:
                continue
        
        # Filter by area and position
        if area > min_area and x > 5 and (x + w) < (width - 5):
            cnt_center_x = x + w // 2
            distance_from_center = abs(cnt_center_x - center_x)
            
            # Check if contour touches substrate
            if substrate_y is not None:
                distance_to_substrate = abs(bottom_y - substrate_y)
                touches_substrate = distance_to_substrate <= cfg.substrate_touch_tolerance
                
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
        # Fallback to largest contour
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > min_area:
            hull = cv2.convexHull(largest)
            return hull[:, 0, :].astype(np.float64), None
        return None
    
    # Select best contour
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
    **kwargs
) -> Optional[np.ndarray]:
    """
    Detect drop contour for pendant drop using Otsu thresholding.
    """
    import cv2 

    raw_cfg = resolve_plugin_settings("pendant", kwargs.get("plugin_settings", {}), **kwargs)
    cfg = DropPendantSettings(**raw_cfg)

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
    min_area = image_area * cfg.min_area_fraction
    
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
            if abs(cx - img_center_x) < (width * cfg.centering_tolerance):
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

# Register settings
# Reuse "sessile" and "pendant" keys within this file scope
register_detector_settings("sessile", DropSessileSettings) 
register_detector_settings("pendant", DropPendantSettings)

