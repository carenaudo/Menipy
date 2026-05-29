"""
Drop contour detection plugin using auto-calibration algorithms.

Provides automatic drop contour detection for droplet analysis.
"""

from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from menipy.common.plugin_settings import (
    register_detector_settings,
    resolve_plugin_settings,
)
from menipy.common.registry import register_drop_detector
from menipy.common.sessile_detection import detect_sessile_drop_contour

# NOTE: cv2 import moved inside functions


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration Models
# -----------------------------------------------------------------------------


class DropSessileSettings(BaseModel):
    """Settings for sessile drop detection.

    Configuration parameters for controlling the adaptive thresholding-based
    sessile drop contour detection algorithm.
    """

    model_config = ConfigDict(extra="ignore")

    clahe_clip_limit: float = Field(2.0, description="Contrast enhancement limit")
    adaptive_block_size: int = Field(21, description="Threshold block size (odd)")
    adaptive_c: int = Field(2, description="Threshold constant")
    min_area_fraction: float = Field(0.005, description="Min area as fraction of image")
    substrate_touch_tolerance: int = Field(
        15, description="Max pixels from substrate to be considered touching"
    )
    rectangularity_threshold: float = Field(
        0.85, description="Max rectangularity to filter out ROI boxes"
    )
    min_gap_from_needle: int = Field(
        40, description="Min vertical gap from needle bottom for sessile contour"
    )
    needle_alignment_guard: int = Field(
        100, description="Max vertical distance for center-alignment needle guard"
    )

    def model_post_init(self, __context):
        """Ensure adaptive block size is odd for cv2.adaptiveThreshold."""
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1


class DropPendantSettings(BaseModel):
    """Settings for pendant drop detection.

    Configuration parameters for controlling the Otsu thresholding-based
    pendant drop contour detection algorithm.
    """

    model_config = ConfigDict(extra="ignore")

    min_area_fraction: float = Field(0.05, description="Min area as fraction of image")
    centering_tolerance: float = Field(
        0.3, description="Max horizontal offset from center (fraction of width)"
    )


# -----------------------------------------------------------------------------
# Implementations
# -----------------------------------------------------------------------------


def detect_drop_sessile(
    image: np.ndarray,
    *,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: tuple[int, int] = (8, 8),
    adaptive_block_size: int = 21,
    adaptive_c: int = 2,
    substrate_y: int | None = None,
    needle_rect: tuple[int, int, int, int] | None = None,
    min_area_fraction: float = 0.005,
    **kwargs,
) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]] | None:
    """Detect drop contour for sessile drops using adaptive thresholding.

    This function detects the sessile drop contour by applying CLAHE contrast
    enhancement followed by adaptive thresholding. The algorithm includes
    morphological filtering to remove noise and substrate masking.
    Detected contours are filtered by area and spatial criteria.

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
    substrate_y : Optional[int], optional
        Y-coordinate of substrate line. If provided, image rows below
        substrate_y-2 are masked to avoid detecting substrate.
        Default is None.
    min_area_fraction : float, optional
        Minimum contour area as a fraction of image area. Default is 0.005.
    **kwargs : dict
        Additional keyword arguments including plugin_settings dictionary.

    Returns
    -------
    Optional[Tuple[np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]]]]
        Tuple containing:
        - Drop contour as array of points
        - Contact points as ((x_left, y_substrate), (x_right, y_substrate))
          or None if no substrate was provided

        Returns None if no valid contour is detected.

    Raises
    ------
    None
        Returns None instead of raising exceptions for invalid input.

    Notes
    -----
    Contour filtering logic:
    1. Rejects contours touching image border or touching top (needle)
    2. Filters by area (must exceed min_area_fraction)
    3. Prefers contours touching substrate when available
    4. Applies convex hull and reconstructs with substrate baseline

    See Also
    --------
    detect_drop_pendant : Detects drop contour for pendant drops
    """
    raw_cfg = resolve_plugin_settings(
        "sessile", kwargs.get("plugin_settings", {}), **kwargs
    )
    cfg = DropSessileSettings(**raw_cfg)

    # Accept legacy callers that pass geometry via kwargs/plugin settings wrapper.
    if needle_rect is None:
        maybe_needle = kwargs.get("needle_rect")
        if isinstance(maybe_needle, tuple) and len(maybe_needle) == 4:
            needle_rect = maybe_needle

    detection = detect_sessile_drop_contour(
        image,
        substrate_y=substrate_y,
        needle_rect=needle_rect,
        min_area_fraction=cfg.min_area_fraction,
        substrate_touch_tolerance=cfg.substrate_touch_tolerance,
        rectangularity_threshold=cfg.rectangularity_threshold,
        min_gap_from_needle=cfg.min_gap_from_needle,
        needle_alignment_guard=cfg.needle_alignment_guard,
        clahe_clip_limit=cfg.clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        adaptive_block_size=cfg.adaptive_block_size,
        adaptive_c=cfg.adaptive_c,
    )
    if detection.contour is None:
        return None

    logger.info("Sessile drop detected with %s points", len(detection.contour))
    return detection.contour, detection.contact_points


def detect_drop_pendant(
    image: np.ndarray, *, min_area_fraction: float = 0.05, **kwargs
) -> np.ndarray | None:
    """Detect drop contour for pendant drops using Otsu thresholding.

    This function detects the drop contour in pendant drop analysis by
    applying Otsu's automatic thresholding method. The contour must satisfy
    area and centering constraints to be considered valid. The detected
    contour is post-processed with morphological operations to remove noise.

    Parameters
    ----------
    image : np.ndarray
        Input image as a 2D array (grayscale) or 3D array (BGR color).
        If color image is provided, it is automatically converted to grayscale.
    min_area_fraction : float, optional
        Minimum required contour area as a fraction of the total image area.
        Contours smaller than this value are rejected. Default is 0.05.
    **kwargs : dict
        Additional keyword arguments, including:
        - plugin_settings : dict, optional
            Configuration dictionary for detector settings.

    Returns
    -------
    Optional[np.ndarray]
        Drop contour as an array of contour points, or None if no valid
        contour is detected. Contour format typically follows OpenCV
        convention with shape (N, 1, 2) where N is the number of points.

    Raises
    ------
    None
        Returns None instead of raising exceptions for invalid input.

    Notes
    -----
    A valid contour must meet all of the following criteria:
    - Area exceeds min_area_fraction * image_area
    - Centroid is within width * centering_tolerance pixels of image center

    The function applies morphological opening to remove small noise artifacts
    and uses Otsu's method for automatic threshold selection, making it robust
    to varying image contrast levels.

    See Also
    --------
    detect_drop_sessile : Detects drop contour for sessile drops
    """
    import cv2

    raw_cfg = resolve_plugin_settings(
        "pendant", kwargs.get("plugin_settings", {}), **kwargs
    )
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
