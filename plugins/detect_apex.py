"""
Apex detection plugin for pendant drop analysis.

Provides automatic apex point detection for pendant drop analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from menipy.common.registry import register_apex_detector
from menipy.math.apex import detect_apex

logger = logging.getLogger(__name__)


def detect_apex_pendant(
    drop_contour: np.ndarray,
    **kwargs: Any,
) -> tuple[int, int] | None:
    """Detect the apex point for pendant drops using centroid-averaged sub-pixel refinement.

    Parameters
    ----------
    drop_contour : np.ndarray
        Drop contour array as Nx2 or Nx1x2 shape.
    **kwargs : dict
        Additional parameters passed to detect_apex (e.g. refine, window_px).

    Returns
    -------
    Optional[Tuple[int, int]]
        Apex point as (x, y) tuple with integer coordinates, or None.
    """
    if drop_contour is None or len(drop_contour) == 0:
        logger.warning("Apex detection requires drop contour")
        return None

    try:
        res = detect_apex(drop_contour, mode="pendant", **kwargs)
        apex = res.to_int_tuple()
        logger.info(f"Pendant apex detected at {apex} via {res.method} (conf={res.confidence:.2f})")
        return apex
    except Exception as e:
        logger.warning(f"Pendant apex detection error: {e}")
        return None


def detect_apex_sessile(
    drop_contour: np.ndarray,
    substrate_y: int | None = None,
    baseline: tuple[tuple[float, float], tuple[float, float]] | None = None,
    substrate: Any = None,
    contact_points: tuple[tuple[float, float], tuple[float, float]] | None = None,
    **kwargs: Any,
) -> tuple[int, int] | None:
    """Detect the apex point for sessile drops, supporting tilted baselines and sub-pixel refinement.

    Parameters
    ----------
    drop_contour : np.ndarray
        Drop contour array as Nx2 or Nx1x2 shape.
    substrate_y : Optional[int], optional
        Y-coordinate of flat horizontal substrate line.
    baseline : Optional[Tuple], optional
        Substrate baseline chord ((x1, y1), (x2, y2)) for tilted substrates.
    substrate : Optional[Any], optional
        SubstrateProfile object for curved substrates.
    contact_points : Optional[Tuple], optional
        Three-phase contact points.
    **kwargs : dict
        Additional parameters passed to detect_apex.

    Returns
    -------
    Optional[Tuple[int, int]]
        Apex point as (x, y) tuple with integer coordinates, or None.
    """
    if drop_contour is None or len(drop_contour) == 0:
        logger.warning("Apex detection requires drop contour")
        return None

    effective_baseline = baseline
    if effective_baseline is None and substrate_y is not None:
        effective_baseline = ((0.0, float(substrate_y)), (10000.0, float(substrate_y)))

    try:
        res = detect_apex(
            drop_contour,
            mode="sessile",
            baseline=effective_baseline,
            substrate=substrate,
            contact_points=contact_points,
            **kwargs,
        )
        apex = res.to_int_tuple()
        logger.info(f"Sessile apex detected at {apex} via {res.method} (conf={res.confidence:.2f})")
        return apex
    except Exception as e:
        logger.warning(f"Sessile apex detection error: {e}")
        return None


def detect_apex_auto(
    drop_contour: np.ndarray, pipeline: str = "sessile", **kwargs: Any
) -> tuple[int, int] | None:
    """Auto-detect apex based on the analysis pipeline type."""
    if pipeline.lower() in ("pendant", "captive_bubble"):
        return detect_apex_pendant(drop_contour, **kwargs)
    else:
        return detect_apex_sessile(drop_contour, **kwargs)


# Register plugins
register_apex_detector("pendant", detect_apex_pendant)
register_apex_detector("sessile", detect_apex_sessile)
register_apex_detector("captive_bubble", detect_apex_pendant)
register_apex_detector("auto", detect_apex_auto)
