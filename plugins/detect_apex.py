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
    """
    Detect apex point for pendant drop (bottom of drop).
    
    The apex is defined as the point with maximum Y coordinate
    (lowest point in the image coordinate system).
    
    Args:
        drop_contour: Drop contour as Nx2 or Nx1x2 array
        
    Returns:
        Apex point as (x, y) or None if contour is invalid.
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
    """
    Detect apex point for sessile drop (top of drop dome).
    
    The apex is defined as the point with minimum Y coordinate
    (highest point in the image coordinate system), excluding
    points near the substrate.
    
    Args:
        drop_contour: Drop contour as Nx2 or Nx1x2 array
        substrate_y: Y-coordinate of substrate (to exclude base points)
        
    Returns:
        Apex point as (x, y) or None if contour is invalid.
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
    """
    Auto-detect apex based on pipeline type.
    
    Args:
        drop_contour: Drop contour array
        pipeline: Pipeline name ("sessile" or "pendant")
        **kwargs: Additional parameters
        
    Returns:
        Apex point as (x, y) or None.
    """
    if pipeline.lower() == "pendant":
        return detect_apex_pendant(drop_contour)
    else:
        return detect_apex_sessile(drop_contour, **kwargs)


# Register plugins
register_apex_detector("pendant", detect_apex_pendant)
register_apex_detector("sessile", detect_apex_sessile)
register_apex_detector("auto", detect_apex_auto)
