"""
Apex detection plugin for pendant drop analysis.

Provides automatic apex point detection for pendant drop analysis.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from menipy.common.registry import register_apex_detector

logger = logging.getLogger(__name__)


def detect_apex_pendant(
    drop_contour: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """Detect the apex point for pendant drops as the lowest point.
    
    The apex is defined as the point with maximum Y coordinate,
    which represents the lowest point in the image coordinate system.
    This point is critical for pendant drop analysis as it defines
    the reference for drop geometry calculations.
    
    Parameters
    ----------
    drop_contour : np.ndarray
        Drop contour array as Nx2 or Nx1x2 shape, where N is the number
        of contour points and columns represent x and y coordinates.
    
    Returns
    -------
    Optional[Tuple[int, int]]
        Apex point as (x, y) tuple with integer coordinates, or None if
        the contour is invalid or empty.
    
    Raises
    ------
    None
        Returns None instead of raising exceptions for invalid input.
    
    Examples
    --------
    >>> contour = np.array([[10, 20], [15, 30], [20, 25]])
    >>> apex = detect_apex_pendant(contour)
    >>> print(apex)
    (15, 30)
    
    Notes
    -----
    The function automatically reshapes Nx1x2 contours to Nx2 format.
    Empty contours return None with a warning log message.
    """
    if drop_contour is None or len(drop_contour) == 0:
        logger.warning("Apex detection requires drop contour")
        return None
    
    # Reshape if needed
    pts = np.asarray(drop_contour)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    
    if len(pts) == 0:
        return None
    
    # Apex is the point with maximum Y (bottom of drop)
    apex_idx = np.argmax(pts[:, 1])
    apex_pt = pts[apex_idx]
    
    apex = (int(apex_pt[0]), int(apex_pt[1]))
    logger.info(f"Pendant apex detected at {apex}")
    return apex


def detect_apex_sessile(
    drop_contour: np.ndarray,
    substrate_y: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Detect the apex point for sessile drops as the highest point.
    
    The apex is defined as the point with minimum Y coordinate, which
    represents the highest point in the image coordinate system. This
    corresponds to the top of the drop dome and is essential for
    contact angle calculations. Points near the substrate are excluded
    to ensure the apex is not at the three-phase contact line.
    
    Parameters
    ----------
    drop_contour : np.ndarray
        Drop contour array as Nx2 or Nx1x2 shape, where N is the number
        of contour points and columns represent x and y coordinates.
    substrate_y : Optional[int], optional
        Y-coordinate of the substrate line. If provided, points within
        5 pixels of the substrate are excluded from apex detection.
        Default is None (no substrate filtering).
    
    Returns
    -------
    Optional[Tuple[int, int]]
        Apex point as (x, y) tuple with integer coordinates, or None if
        the contour is invalid, empty, or all points are near substrate.
    
    Raises
    ------
    None
        Returns None instead of raising exceptions for invalid input.
    
    Examples
    --------
    >>> contour = np.array([[15, 10], [20, 5], [25, 10]])
    >>> apex = detect_apex_sessile(contour)
    >>> print(apex)
    (20, 5)
    
    Notes
    -----
    The function automatically reshapes Nx1x2 contours to Nx2 format.
    When substrate_y is provided, contour points are filtered to exclude
    y-coordinates >= substrate_y - 5 pixels.
    """
    if drop_contour is None or len(drop_contour) == 0:
        logger.warning("Apex detection requires drop contour")
        return None
    
    # Reshape if needed
    pts = np.asarray(drop_contour)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    
    if len(pts) == 0:
        return None
    
    # Filter out points near substrate if provided
    if substrate_y is not None:
        pts = pts[pts[:, 1] < (substrate_y - 5)]
        if len(pts) == 0:
            return None
    
    # Apex is the point with minimum Y (top of dome)
    apex_idx = np.argmin(pts[:, 1])
    apex_pt = pts[apex_idx]
    
    apex = (int(apex_pt[0]), int(apex_pt[1]))
    logger.info(f"Sessile apex detected at {apex}")
    return apex


def detect_apex_auto(
    drop_contour: np.ndarray,
    pipeline: str = "sessile",
    **kwargs
) -> Optional[Tuple[int, int]]:
    """Auto-detect apex based on the analysis pipeline type.
    
    This function serves as a dispatcher that calls the appropriate
    apex detection function (pendant or sessile) based on the specified
    pipeline parameter. It provides a unified interface for apex detection
    across different drop analysis modes.
    
    Parameters
    ----------
    drop_contour : np.ndarray
        Drop contour array as Nx2 or Nx1x2 shape, where N is the number
        of contour points and columns represent x and y coordinates.
    pipeline : str, optional
        Pipeline type to determine detection method. Must be one of:
        - "pendant": Uses detect_apex_pendant (apex is lowest point)
        - "sessile" or any other value: Uses detect_apex_sessile
          (apex is highest point after substrate filtering)
        Default is "sessile".
    **kwargs : dict
        Additional keyword arguments passed to the selected detection
        function. For sessile detection, may include substrate_y.
    
    Returns
    -------
    Optional[Tuple[int, int]]
        Apex point as (x, y) tuple with integer coordinates, or None if
        apex detection fails for the selected pipeline.
    
    Raises
    ------
    None
        Returns None instead of raising exceptions.
    
    Examples
    --------
    >>> contour = np.array([[10, 20], [15, 30], [20, 25]])
    >>> apex_pendant = detect_apex_auto(contour, pipeline="pendant")
    >>> apex_sessile = detect_apex_auto(contour, pipeline="sessile")
    
    See Also
    --------
    detect_apex_pendant : Detects apex for pendant drops
    detect_apex_sessile : Detects apex for sessile drops
    """
    if pipeline.lower() == "pendant":
        return detect_apex_pendant(drop_contour)
    else:
        return detect_apex_sessile(drop_contour, **kwargs)


# Register plugins
register_apex_detector("pendant", detect_apex_pendant)
register_apex_detector("sessile", detect_apex_sessile)
register_apex_detector("auto", detect_apex_auto)
