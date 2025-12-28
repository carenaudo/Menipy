"""
Minimal preprocessing stage for the pendant pipeline.
Converts image to grayscale and applies a small Gaussian blur, setting
`ctx.preprocessed` and basic metadata so downstream stages have a stable baseline.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import cv2

from menipy.models.context import Context


def run(ctx: Context) -> Optional[Context]:
    img = getattr(ctx, "image", None)
    if img is None:
        return ctx

    try:
        if hasattr(img, "ndim") and img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
    except Exception:
        return ctx

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ctx.preprocessed = blurred
    ctx.preprocessed_settings = {"blur_ksize": (5, 5)}
    ctx.preprocessed_history = ["gaussian_blur"]
    ctx.preprocessed_roi = (0, 0, int(blurred.shape[1]), int(blurred.shape[0]))
    return ctx
