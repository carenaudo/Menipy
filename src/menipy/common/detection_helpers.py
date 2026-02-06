"""
Detection stage helpers for pipeline integration.

This module provides pipeline-agnostic helper functions that use the
detection plugins to automatically detect ROI, needle, drop, and other
features from images.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np

from menipy.common.image_utils import ensure_gray, edges_to_xy

logger = logging.getLogger(__name__)

# Import detection plugins (registers them with the registry)
try:
    import sys
    from pathlib import Path
    plugins_dir = Path(__file__).parent.parent.parent.parent / "plugins"
    if plugins_dir.exists() and str(plugins_dir) not in sys.path:
        sys.path.insert(0, str(plugins_dir))
    
    import detect_needle
    import detect_roi
    import detect_substrate
    import detect_drop
    import detect_apex
    _PLUGINS_LOADED = True
except ImportError as e:
    logger.warning(f"Detection plugins not available: {e}")
    _PLUGINS_LOADED = False

from menipy.common.registry import (
    NEEDLE_DETECTORS,
    ROI_DETECTORS,
    SUBSTRATE_DETECTORS,
    DROP_DETECTORS,
    APEX_DETECTORS,
)


def plugins_available() -> bool:
    """Check if detection plugins are loaded."""
    return _PLUGINS_LOADED and len(NEEDLE_DETECTORS) > 0


def auto_detect_features(
    image: np.ndarray,
    pipeline: str = "sessile",
    *,
    detect_needle: bool = True,
    detect_substrate: bool = True,
    detect_drop: bool = True,
    detect_apex: bool = True,
    detect_roi: bool = True,
) -> Dict[str, Any]:
    """
    Run automatic feature detection for a pipeline.
    
    Args:
        image: Input image (BGR or grayscale)
        pipeline: Pipeline name ("sessile" or "pendant")
        detect_*: Flags to enable/disable specific detections
        
    Returns:
        Dictionary with detected features:
            - needle_rect: (x, y, w, h) or None
        - substrate_line: ((x1, y1), (x2, y2)) or None  
        - drop_contour: Nx2 array or None
        - contact_points: ((left), (right)) or None
        - apex_point: (x, y) or None
        - roi_rect: (x, y, w, h) or None
    """
    if not plugins_available():
        logger.warning("Detection plugins not loaded")
        return {}
    
    results: Dict[str, Any] = {}
    pipeline = pipeline.lower()
    
    # 1. Detect substrate (sessile only)
    substrate_y = None
    if detect_substrate and pipeline == "sessile":
        if "gradient" in SUBSTRATE_DETECTORS:
            substrate_line = SUBSTRATE_DETECTORS["gradient"](image)
            if substrate_line:
                results["substrate_line"] = substrate_line
                # Extract Y for other detections
                substrate_y = int((substrate_line[0][1] + substrate_line[1][1]) / 2)
    
    # 2. Detect needle
    if detect_needle:
        detector_name = pipeline if pipeline in NEEDLE_DETECTORS else "sessile"
        if detector_name in NEEDLE_DETECTORS:
            needle_result = NEEDLE_DETECTORS[detector_name](image)
            if needle_result:
                if pipeline == "pendant" and isinstance(needle_result, tuple) and len(needle_result) == 2:
                    # Pendant returns (needle_rect, contact_points)
                    results["needle_rect"] = needle_result[0]
                    results["contact_points"] = needle_result[1]
                else:
                    results["needle_rect"] = needle_result
    
    # 3. Detect drop contour
    drop_contour = None
    if detect_drop:
        detector_name = pipeline if pipeline in DROP_DETECTORS else "sessile"
        if detector_name in DROP_DETECTORS:
            if pipeline == "sessile":
                drop_result = DROP_DETECTORS[detector_name](image, substrate_y=substrate_y)
                if drop_result:
                    if isinstance(drop_result, tuple) and len(drop_result) == 2:
                        drop_contour, contact_pts = drop_result
                        results["drop_contour"] = drop_contour
                        if contact_pts:
                            results["contact_points"] = contact_pts
                    else:
                        drop_contour = drop_result
                        results["drop_contour"] = drop_contour
            else:
                drop_contour = DROP_DETECTORS[detector_name](image)
                if drop_contour is not None:
                    results["drop_contour"] = drop_contour
    
    # 4. Detect apex
    apex_point = None
    if detect_apex and drop_contour is not None:
        detector_name = pipeline if pipeline in APEX_DETECTORS else "auto"
        if detector_name in APEX_DETECTORS:
            if pipeline == "sessile":
                apex_point = APEX_DETECTORS[detector_name](drop_contour, substrate_y=substrate_y)
            else:
                apex_point = APEX_DETECTORS[detector_name](drop_contour)
            if apex_point:
                results["apex_point"] = apex_point
    
    # 5. Detect ROI
    if detect_roi:
        detector_name = pipeline if pipeline in ROI_DETECTORS else "auto"
        if detector_name in ROI_DETECTORS:
            # Build pipeline-specific kwargs
            if pipeline == "sessile":
                roi_kwargs = {
                    "drop_contour": results.get("drop_contour"),
                    "substrate_y": substrate_y,
                    "needle_rect": results.get("needle_rect"),
                }
            else:  # pendant
                roi_kwargs = {
                    "drop_contour": results.get("drop_contour"),
                    "apex_point": results.get("apex_point"),
                }
            
            roi_rect = ROI_DETECTORS[detector_name](image, **roi_kwargs)
            if roi_rect:
                results["roi_rect"] = roi_rect
    
    logger.info(f"Auto-detected features for {pipeline}: {list(results.keys())}")
    return results


def apply_roi_crop(
    image: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Crop image to ROI region.
    
    Args:
        image: Input image
        roi_rect: (x, y, width, height)
        
    Returns:
        Cropped image.
    """
    x, y, w, h = roi_rect
    return image[y:y+h, x:x+w].copy()


def adjust_coordinates_for_roi(
    points: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Adjust point coordinates for ROI offset.
    
    Args:
        points: Nx2 array of (x, y) coordinates
        roi_rect: (x, y, width, height)
        
    Returns:
        Adjusted points with ROI offset subtracted.
    """
    x, y, w, h = roi_rect
    adjusted = points.copy()
    adjusted[:, 1] -= y
    return adjusted


# -----------------------------------------------------------------------------
# Shared Edge Detection Helpers 
# -----------------------------------------------------------------------------
# Re-exporting from image_utils for backward compatibility if needed
# (though ideally consumers should import from image_utils directly)
__all__ = [
    "ensure_gray",
    "edges_to_xy",
    "auto_detect_features",
    "apply_roi_crop",
    "adjust_coordinates_for_roi",
    "plugins_available",
]
