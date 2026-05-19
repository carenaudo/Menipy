"""
Drop contour detection preprocessor plugin.

This plugin detects the drop contour and stores it in the context.
Follows the stage-based pattern: operates on ctx and returns ctx.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

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

    # Delegate to detector plugins that are aligned to AutoCalibrator baseline.
    try:
        import detect_drop
    except Exception:
        logger.exception("detect_drop plugin is unavailable")
        return ctx

    if pipeline == "pendant":
        contour = detect_drop.detect_drop_pendant(image)
        if contour is not None and len(contour) > 0:
            ctx.detected_contour = contour
            try:
                pts = contour.reshape(-1, 2)
                apex_idx = np.argmax(pts[:, 1])
                ctx.apex_point = (int(pts[apex_idx][0]), int(pts[apex_idx][1]))
            except Exception:
                pass
    else:
        substrate_line = getattr(ctx, "substrate_line", None)
        substrate_y = None
        if substrate_line:
            substrate_y = int((substrate_line[0][1] + substrate_line[1][1]) / 2)

        needle_rect = getattr(ctx, "needle_rect", None)
        substrate_touch_tolerance = getattr(ctx, "substrate_touch_tolerance_px", None)

        kwargs = {
            "substrate_y": substrate_y,
            "needle_rect": needle_rect,
        }
        if substrate_touch_tolerance is not None:
            kwargs["substrate_touch_tolerance"] = int(substrate_touch_tolerance)

        result = detect_drop.detect_drop_sessile(image, **kwargs)
        if result is not None:
            if isinstance(result, tuple) and len(result) == 2:
                contour, contact_pts = result
                if contour is not None and len(contour) > 0:
                    ctx.detected_contour = contour
                if contact_pts is not None:
                    ctx.contact_points = contact_pts
            else:
                contour = result
                if contour is not None and len(contour) > 0:
                    ctx.detected_contour = contour

    if getattr(ctx, "detected_contour", None) is not None:
        logger.info("Drop detected for %s pipeline", pipeline)

    return ctx


# Register as preprocessor plugin
register_preprocessor("detect_drop", detect_drop_preprocessor)
