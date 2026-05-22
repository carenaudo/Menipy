"""
Needle detection preprocessor plugin.

This plugin detects the needle region and stores it in the context.
Follows the stage-based pattern: operates on ctx and returns ctx.
"""

from __future__ import annotations

import logging

import numpy as np

from menipy.common.registry import register_preprocessor

logger = logging.getLogger(__name__)


def detect_needle_preprocessor(ctx):
    """
    Preprocessor plugin that detects needle region.

    For sessile: finds contour touching top border.
    For pendant: uses shaft line analysis.

    Stores result in ctx.needle_rect as (x, y, width, height).

    Args:
        ctx: Pipeline context with image data

    Returns:
        Updated context with needle_rect set.
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

    # Delegate to detector plugins aligned with AutoCalibrator baseline.
    try:
        import detect_needle
    except Exception:
        logger.exception("detect_needle plugin is unavailable")
        return ctx

    if pipeline == "pendant":
        drop_contour = getattr(ctx, "detected_contour", None)
        if drop_contour is None:
            # Fallback: reuse drop detector plugin to produce contour for needle analysis.
            try:
                import detect_drop

                drop_contour = detect_drop.detect_drop_pendant(image)
            except Exception:
                drop_contour = None

        if drop_contour is not None:
            result = detect_needle.detect_needle_pendant(
                image, drop_contour=drop_contour
            )
            if result is not None:
                needle_rect, contact_points = result
                ctx.needle_rect = needle_rect
                ctx.contact_points = contact_points
                logger.info("Pendant needle detected: %s", needle_rect)
    else:
        needle_rect = detect_needle.detect_needle_sessile(image)
        if needle_rect is not None:
            ctx.needle_rect = needle_rect
            logger.info("Sessile needle detected: %s", needle_rect)

    return ctx


# Register as preprocessor plugin
register_preprocessor("detect_needle", detect_needle_preprocessor)
