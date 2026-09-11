"""Shared utilities for pipeline stages."""

from __future__ import annotations

import numpy as np

from menipy.common import edge_detection as edged
from menipy.models.config import EdgeDetectionSettings
from menipy.models.context import Context


def ensure_contour(
    ctx: Context, settings: EdgeDetectionSettings | None = None
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
    if ctx.contour is not None and ctx.contour.xy is not None:
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
            ctx.frames = list(loaded)
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
    if ctx.contour is not None and ctx.contour.xy is not None:
        return np.asarray(ctx.contour.xy, dtype=float)

    return np.empty((0, 2), dtype=float)


def image_for_contour(ctx: Context, contour_xy: np.ndarray) -> np.ndarray | None:
    """Return the image the contour was measured on, or ``None`` if unavailable.

    Sub-pixel edge refinement samples intensity profiles at contour
    coordinates, so an image is only returned when every contour point lies
    inside it -- a cropped or rescaled frame would put the samples in the wrong
    place.

    Parameters
    ----------
    ctx : Context
        Pipeline context holding ``image``, ``frames`` or ``image_path``.
    contour_xy : np.ndarray
        Contour in image coordinates, shape ``(N, 2)``.

    Returns
    -------
    np.ndarray or None
        The image, or ``None`` when none matches the contour frame.
    """
    image = getattr(ctx, "image", None)
    if image is not None and not isinstance(image, np.ndarray):
        image = getattr(image, "image", None)
    if image is None:
        frames = getattr(ctx, "frames", None) or []
        first = frames[0] if len(frames) else None
        image = getattr(first, "image", first) if first is not None else None
    if image is None and getattr(ctx, "image_path", None):
        import cv2

        image = cv2.imread(str(ctx.image_path), cv2.IMREAD_UNCHANGED)
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        return None
    xy = np.asarray(contour_xy, dtype=float).reshape(-1, 2)
    h, w = image.shape[:2]
    if xy.size == 0 or xy[:, 0].min() < 0 or xy[:, 1].min() < 0 or xy[:, 0].max() > w - 1 or xy[:, 1].max() > h - 1:
        return None
    return image
