# src/menipy/common/preprocessing.py
from __future__ import annotations
import numpy as np

def run(ctx):
    """Basic grayscale→binary placeholder; replace with real denoise/threshold."""
    frames = ctx.frames if isinstance(ctx.frames, list) else [ctx.frames]
    # TODO: cv2.GaussianBlur + cv2.threshold / adaptive threshold
    ctx.frames = [np.clip(f, 0, 255).astype(np.uint8) for f in frames]
    return ctx
