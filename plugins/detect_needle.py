"""
Needle detection plugin using auto-calibration algorithms.

Provides needle detection for both sessile and pendant pipelines.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
# NOTE: cv2 import moved inside functions

from menipy.common.registry import register_needle_detector
from pydantic import BaseModel, Field, ConfigDict
from menipy.common.plugin_settings import register_detector_settings, resolve_plugin_settings

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration Models
# -----------------------------------------------------------------------------

class NeedleSessileSettings(BaseModel):
    """Settings for sessile drop needle detection.
    
    Configuration parameters for controlling the adaptive thresholding-based
    needle detection algorithm for sessile drops.
    """

    model_config = ConfigDict(extra='ignore')
    
    clahe_clip_limit: float = Field(2.0, description="Contrast enhancement limit")
    adaptive_block_size: int = Field(21, description="Threshold block size (must be odd)")
    adaptive_c: int = Field(2, description="Threshold constant")

    def model_post_init(self, __context):
        """Ensure adaptive block size is odd for cv2.adaptiveThreshold."""
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1

class NeedlePendantSettings(BaseModel):
    """Settings for pendant drop needle detection.
    
    Configuration parameters for controlling the shaft line analysis-based
    needle detection algorithm for pendant drops.
    """

    model_config = ConfigDict(extra='ignore')
    
    tolerance: int = Field(3, description="Pixel tolerance for shaft straightness")
    top_limit_offset: int = Field(20, description="Height from top to scan for shaft")


# -----------------------------------------------------------------------------
# Implementations
# -----------------------------------------------------------------------------

def detect_needle_sessile(
    image: np.ndarray,
    *,
    clahe_clip_limit: float = 2.0, # Keeps kwargs for backward compatibility
    clahe_tile_size: Tuple[int, int] = (8, 8),
    adaptive_block_size: int = 21,
    adaptive_c: int = 2,
    **kwargs
) -> Optional[Tuple[int, int, int, int]]:
    """Detect needle region in sessile drop images as contour touching top.
    
    This function detects the needle in sessile drop images by identifying
    the contour that touches the top border of the image. The detection uses
    adaptive thresholding with contrast-limited histogram equalization (CLAHE)
    to enhance needle visibility.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as a 2D array (grayscale) or 3D array (BGR color).
        If color image is provided, it is automatically converted to grayscale.
    clahe_clip_limit : float, optional
        CLAHE clip limit for contrast enhancement. Default is 2.0.
    clahe_tile_size : Tuple[int, int], optional
        CLAHE tile grid size as (height, width). Default is (8, 8).
    adaptive_block_size : int, optional
        Threshold block size for adaptive thresholding (must be odd).
        Default is 21.
    adaptive_c : int, optional
        Threshold constant for adaptive thresholding. Default is 2.
    **kwargs : dict
        Additional keyword arguments including plugin_settings dictionary.
    
    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        Needle bounding box as (x, y, width, height) where the contour
        touches the top border of the image. Returns None if no such
        contour is detected.
    
    Raises
    ------
    None
        Returns None instead of raising exceptions for invalid input.
    
    Notes
    -----
    The needle is identified as the first contour found in the filtered binary
    image that has y-coordinate (top edge) less than 5 pixels from the image
    top border. Morphological operations are applied to clean the binary image
    before contour detection.
    
    See Also
    --------
    detect_needle_pendant : Detects needle for pendant drops
    """
    import cv2 

    # Resolve settings
    raw_cfg = resolve_plugin_settings("sessile_needle_detect", kwargs.get("plugin_settings", {}), **kwargs)
    # We use local variables as defaults if not in settings, but Pydantic defaults take precedence if instantiating fresh
    # Ideally, we map kwargs to the model. 
    # To respect passed-in args vs defaults:
    
    # 1. Start with values from args (which might hold defaults from signature)
    # But since Pydantic holds the source of truth for defaults now, we prefer that.
    # However, existing callers might pass arguments positionally or by keyword.
    
    # Let's trust resolve_plugin_settings to handle dicts
    # We will override the signature defaults with what the settings say, UNLESS explicitly passed?
    # Simpler: just use the settings model.
    
    # But wait, 'sessile' is the registered name for the needle detector in 'needle_detectors' registry?
    # Below: register_needle_detector("sessile", ...)
    # So we should key config on "sessile" (or "needle_detectors.sessile"?)
    # For now, let's use a unique name "needle_sessile" for config registry to avoid collision if any
    
    # Actually, let's look at how we register: register_detector_settings("needle_sessile", ...)
    
    # Note: `resolve_plugin_settings` expects a dict.
    # We construct a dict from kwargs, but prioritize explicit args if they differ from python defaults?
    # That's hard to detect. 
    # Let's just assume `plugin_settings` dict is the primary source of configuration 
    # and explicit kwargs are overrides.
    
    cfg = NeedleSessileSettings(**raw_cfg)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
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
    **kwargs
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Detect needle region and contact points for pendant drops.
    
    This function analyzes the pendant drop contour to locate the needle
    shaft and determine the contact points where the drop touches the needle.
    The detection is based on analyzing vertical contour positions within
    defined tolerance limits to identify where the drop boundary deviates
    from a straight needle shaft.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as a 2D array (grayscale) or 3D array (BGR color).
        Used for contour visualization and validation.
    drop_contour : Optional[np.ndarray], optional
        Pre-detected drop contour from drop detection stage. Required for
        identifying the needle shaft location from the drop extension.
        Default is None.
    tolerance : int, optional
        Pixel tolerance for detecting deviation from the straight shaft.
        Points within this tolerance are considered part of the shaft.
        Default is 3 pixels.
    **kwargs : dict
        Additional keyword arguments, including:
        - plugin_settings : dict, optional
            Configuration dictionary with detector settings.
    
    Returns
    -------
    Optional[Tuple[Tuple[int, int, int, int], Tuple[Tuple[int, int], Tuple[int, int]]]]
        Tuple containing:
        - Needle bounding box as (x, y, width, height)
        - Contact points as ((x_left, y_left), (x_right, y_right))
        Returns None if drop_contour is missing or analysis fails.
    
    Raises
    ------
    None
        Returns None instead of raising exceptions for invalid input.
    
    Notes
    -----
    The needle shaft is located by analyzing the upper portion of the drop
    contour (first top_limit_offset pixels) and finding the median x-position
    on each side. Contact points are identified where the drop extends beyond
    the expected shaft width at each vertical level.
    
    The function returns both the needle bounding box for use in ROI detection
    and the precise contact points for interfacial tension calculations.
    
    See Also
    --------
    detect_needle_sessile : Detects needle for sessile drops
    """
    import cv2 

    # Resolve settings
    raw_cfg = resolve_plugin_settings("needle_pendant", kwargs.get("plugin_settings", {}), **kwargs)
    cfg = NeedlePendantSettings(**raw_cfg)

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
    
    # Define needle shaft reference
    top_limit = y + cfg.top_limit_offset
    
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
            if current_x < (ref_x_left - cfg.tolerance):
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
            if current_x > (ref_x_right + cfg.tolerance):
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

# Register settings
# We use names that users will find in the plugin manager dropdown
register_detector_settings("sessile", NeedleSessileSettings) 
register_detector_settings("pendant", NeedlePendantSettings)
# NOTE: The names 'sessile' and 'pendant' are generic. 
# But in PluginManager, they are scoped to this file!
# The user sees: Plugin "detect_needle" -> Configure -> Select "sessile" or "pendant".
# This works perfectly with the previous fix.

