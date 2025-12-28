"""
Minimal preprocessing stage for the sessile pipeline.
This provides a small, well-tested baseline implementation that is safe
and has no external dependencies beyond OpenCV/numpy already used in the project.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import cv2

from menipy.models.context import Context


def run(ctx: Context) -> Optional[Context]:
    """Simple preprocessing that ensures an image is present and produces a blurred grayscale preview.

    - If ctx.image is a path string it is left to higher stages to resolve (acquisition stage).
    - If ctx.image is a numpy array it produces a grayscale blurred image in ctx.preprocessed and sets
      basic preprocessed metadata so downstream stages have a predictable baseline.
    - The stage is intentionally conservative: it never raises and returns the context unchanged when
      image data is absent.
    """
    img = getattr(ctx, "image", None)
    if img is None:
        return ctx

    # If given a color image, convert to grayscale
    try:
        if hasattr(img, "ndim") and img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
    except Exception:
        # Non-standard image - return unchanged
        return ctx

    # Apply a small Gaussian blur to reduce pixel-level noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Populate context fields used by other stages/tests
    ctx.preprocessed = blurred
    ctx.preprocessed_settings = {"blur_ksize": (5, 5)}
    ctx.preprocessed_history = ["gaussian_blur"]
    # Create a basic ROI covering the full image
    ctx.preprocessed_roi = (0, 0, int(blurred.shape[1]), int(blurred.shape[0]))

    return ctx
