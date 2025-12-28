"""
Shared utilities for pipeline stages.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from menipy.models.context import Context
from menipy.models.config import EdgeDetectionSettings
from menipy.common import edge_detection as edged


def ensure_contour(
    ctx: Context, settings: Optional[EdgeDetectionSettings] = None
) -> np.ndarray:
    """
    Get the contour from the context, or run edge detection if it's missing.
    Ensures that the context has loaded frames/images if possible before running detection.

    Args:
        ctx: Pipeline Context
        settings: Optional EdgeDetectionSettings override (uses ctx.edge_detection_settings or defaults to Canny if None)

    Returns:
        (N, 2) numpy array of contour points

    Raises:
        RuntimeError: If no image can be loaded to detect contours.
    """
    # 1. Return existing contour if valid
    if getattr(ctx, "contour", None) is not None and hasattr(ctx.contour, "xy"):
        return np.asarray(ctx.contour.xy, dtype=float)

    # 2. Ensure we have image data in ctx.frames/ctx.image
    frames = getattr(ctx, "frames", None)
    image = getattr(ctx, "image", None)
    image_path = getattr(ctx, "image_path", None)

    # Try to load from image_path if no frames and no image
    if (not frames or len(frames) == 0) and image is None and image_path:
        try:
            from menipy.common import acquisition as acq

            loaded = acq.from_file([image_path])
        except Exception:
            loaded = []
        if loaded:
            ctx.frames = loaded
            ctx.frame = loaded[0]
            ctx.image = loaded[0]

    # If we have an image but no frames, create frames from image
    image = getattr(ctx, "image", None)
    frames = getattr(ctx, "frames", None)
    if image is not None and (not frames or len(frames) == 0):
        from menipy.models.frame import Frame

        if isinstance(image, np.ndarray):
            ctx.frames = [Frame(image=image)]
            ctx.frame = ctx.frames[0]
        elif hasattr(image, "image"):  # Frame object
            ctx.frames = [image]
            ctx.frame = image

    # Final check: ensure we have image data
    frames = getattr(ctx, "frames", None)
    image = getattr(ctx, "image", None)
    if (not frames or len(frames) == 0) and image is None:
        raise RuntimeError(
            "Pipeline: no image available in Context. "
            "Ensure 'acquisition' stage ran first, or provide 'image' or 'image_path' parameter."
        )

    # 3. Run edge detection
    # Use provided settings, or context settings, or default Canny
    run_settings = (
        settings or ctx.edge_detection_settings or EdgeDetectionSettings(method="canny")
    )
    edged.run(ctx, settings=run_settings)

    # 4. Return result
    if ctx.contour and hasattr(ctx.contour, "xy"):
        return np.asarray(ctx.contour.xy, dtype=float)

    return np.empty((0, 2), dtype=float)
