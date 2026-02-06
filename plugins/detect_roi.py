"""
ROI detection plugin using auto-calibration algorithms.

Provides automatic Region of Interest detection for droplet analysis.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from menipy.common.registry import register_roi_detector

logger = logging.getLogger(__name__)


def detect_roi_sessile(
    image: np.ndarray,
    *,
    drop_contour: Optional[np.ndarray] = None,
    substrate_y: Optional[int] = None,
    needle_rect: Optional[Tuple[int, int, int, int]] = None,
    padding: int = 20,
) -> Optional[Tuple[int, int, int, int]]:
    """Detect ROI for sessile drops as a bounding box around drop and substrate.
    
    This function computes the minimal bounding box that encompasses the entire
    drop, substrate line, and optionally the needle region. The ROI is essential
    for focused image processing and analysis, reducing computational overhead
    by limiting processing to the relevant region.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as a 2D array (grayscale) or 3D array (BGR color).
        Used to determine image boundaries for ROI clamping.
    drop_contour : Optional[np.ndarray], optional
        Pre-detected drop contour array in Nx2 or Nx1x2 format. If provided,
        the ROI is extended to include all contour points. Default is None.
    substrate_y : Optional[int], optional
        Y-coordinate of the substrate line. If provided, the ROI is extended
        to include the substrate. Default is None.
    needle_rect : Optional[Tuple[int, int, int, int]], optional
        Needle bounding box as (x, y, width, height). If provided, the lower
        30% of the needle is included in ROI. Default is None.
    padding : int, optional
        Padding in pixels to add around detected regions. Applied after
        determining the initial bounding box boundaries. Default is 20 pixels.
    
    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        ROI as (x, y, width, height) where x and y are the top-left corner
        coordinates and width and height define the ROI dimensions.
        Returns None if insufficient data or ROI dimensions are invalid.
    
    Raises
    ------
    None
        Returns None instead of raising exceptions for invalid input.
    
    Notes
    -----
    At least one of drop_contour, substrate_y, or needle_rect must be provided.
    The ROI is automatically clipped to image boundaries. The function includes
    only the lower 30% of the needle to avoid excessive ROI expansion from the
    entire needle length.
    
    See Also
    --------
    detect_roi_pendant : Detects ROI for pendant drops
    detect_roi_auto : Auto-selects ROI detector by pipeline
    """
    if len(image.shape) == 3:
        height, width = image.shape[:2]
    else:
        height, width = image.shape
    
    x_min, y_min = width, height
    x_max, y_max = 0, 0
    has_data = False
    
    # Include drop contour
    if drop_contour is not None and len(drop_contour) > 0:
        contour = np.asarray(drop_contour)
        if contour.ndim == 3:
            contour = contour.reshape(-1, 2)
        x_min = min(x_min, int(np.min(contour[:, 0])))
        x_max = max(x_max, int(np.max(contour[:, 0])))
        y_min = min(y_min, int(np.min(contour[:, 1])))
        y_max = max(y_max, int(np.max(contour[:, 1])))
        has_data = True
    
    # Include substrate line
    if substrate_y is not None:
        y_max = max(y_max, substrate_y)
        has_data = True
    
    # Include needle (bottom 30%)
    if needle_rect is not None:
        nx, ny, nw, nh = needle_rect
        needle_include_y = ny + int(nh * 0.7)
        y_min = min(y_min, needle_include_y)
        x_min = min(x_min, nx)
        x_max = max(x_max, nx + nw)
        has_data = True
    
    if not has_data:
        logger.warning("ROI detection: no data available")
        return None
    
    # Apply padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    
    if roi_width <= 0 or roi_height <= 0:
        return None
    
    logger.info(f"Sessile ROI detected: ({x_min}, {y_min}, {roi_width}, {roi_height})")
    return (x_min, y_min, roi_width, roi_height)


def detect_roi_pendant(
    image: np.ndarray,
    *,
    drop_contour: Optional[np.ndarray] = None,
    apex_point: Optional[Tuple[int, int]] = None,
    padding: int = 20,
) -> Optional[Tuple[int, int, int, int]]:
    """Detect region of interest (ROI) for pendant drop analysis.
    
    This function computes the minimal bounding box that encompasses the
    entire drop from the needle to the apex point. The ROI is essential for
    focused image processing and analysis, reducing computational overhead
    by limiting processing to the relevant region.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as a 2D array (grayscale) or 3D array (BGR color).
        Used to determine image boundaries for ROI clamping.
    drop_contour : Optional[np.ndarray], optional
        Pre-detected drop contour array in Nx2 or Nx1x2 format. This is
        a required parameter for pendant drop ROI detection.
        Default is None.
    apex_point : Optional[Tuple[int, int]], optional
        Apex point coordinates as (x, y). If provided, the ROI is extended
        to include this point. Default is None.
    padding : int, optional
        Padding in pixels to add around detected regions. Applied after
        determining the initial bounding box boundaries. Default is 20 pixels.
    
    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        ROI as (x, y, width, height) where x and y are the top-left corner
        coordinates and width and height define the ROI dimensions.
        Returns None if drop_contour is missing or ROI dimensions are invalid.
    
    Raises
    ------
    None
        Returns None instead of raising exceptions for invalid input.
    
    Notes
    -----
    The pendant drop ROI extends from the top of the contour to include
    the entire drop and apex. Horizontal padding is applied, but no padding
    is added above the drop minimum y-coordinate (to start from drop edge).
    The ROI is automatically clipped to image boundaries.
    
    See Also
    --------
    detect_roi_sessile : Detects ROI for sessile drops
    detect_roi_auto : Auto-selects ROI detector by pipeline
    """
    if drop_contour is None or len(drop_contour) == 0:
        logger.warning("Pendant ROI detection requires drop contour")
        return None
    
    if len(image.shape) == 3:
        height, width = image.shape[:2]
    else:
        height, width = image.shape
    
    contour = np.asarray(drop_contour)
    if contour.ndim == 3:
        contour = contour.reshape(-1, 2)
    
    x_min = int(np.min(contour[:, 0]))
    x_max = int(np.max(contour[:, 0]))
    y_min = int(np.min(contour[:, 1]))
    y_max = int(np.max(contour[:, 1]))
    
    # Include apex
    if apex_point is not None:
        y_max = max(y_max, apex_point[1])
    
    # Apply padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min)  # Start from top of drop
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    roi_width = x_max - x_min
    roi_height = y_max - y_min
    
    if roi_width <= 0 or roi_height <= 0:
        return None
    
    logger.info(f"Pendant ROI detected: ({x_min}, {y_min}, {roi_width}, {roi_height})")
    return (x_min, y_min, roi_width, roi_height)


def detect_roi_auto(
    image: np.ndarray,
    pipeline: str = "sessile",
    **kwargs
) -> Optional[Tuple[int, int, int, int]]:
    """Auto-detect ROI based on the analysis pipeline type.
    
    This function serves as a dispatcher that calls the appropriate ROI
    detection function (pendant or sessile) based on the specified pipeline
    parameter. It provides a unified interface for ROI detection across
    different drop analysis modes.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as a 2D array (grayscale) or 3D array (BGR color).
        Passed to the selected ROI detection function.
    pipeline : str, optional
        Pipeline type to determine detection method. Must be one of:
        - "pendant": Uses detect_roi_pendant (ROI from needle to apex)
        - "sessile" or any other value: Uses detect_roi_sessile
          (ROI around drop and substrate)
        Default is "sessile".
    **kwargs : dict
        Additional keyword arguments passed to the selected detection
        function. May include drop_contour, apex_point, substrate_y,
        needle_rect, and padding parameters depending on mode.
    
    Returns
    -------
    Optional[Tuple[int, int, int, int]]
        ROI as (x, y, width, height) where x and y are the top-left corner
        coordinates and width and height define the ROI dimensions.
        Returns None if ROI detection fails for the selected pipeline.
    
    Raises
    ------
    None
        Returns None instead of raising exceptions.
    
    Examples
    --------
    >>> roi_sessile = detect_roi_auto(image, pipeline="sessile",
    ...                                drop_contour=contour)
    >>> roi_pendant = detect_roi_auto(image, pipeline="pendant",
    ...                                drop_contour=contour)
    
    See Also
    --------
    detect_roi_pendant : Detects ROI for pendant drops
    detect_roi_sessile : Detects ROI for sessile drops
    """
    if pipeline.lower() == "pendant":
        return detect_roi_pendant(image, **kwargs)
    else:
        return detect_roi_sessile(image, **kwargs)


# Register plugins
register_roi_detector("sessile", detect_roi_sessile)
register_roi_detector("pendant", detect_roi_pendant)
register_roi_detector("auto", detect_roi_auto)
