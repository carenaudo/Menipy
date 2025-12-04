# src/menipy/common/scaling.py
"""
Scaling stage utilities for pixel-to-mm conversion.
"""
from __future__ import annotations

def run(ctx):
    """Attach a simple pixel↔mm scale; pipelines should set real calibration."""
    # Defaults: 1 px = 0.01 mm for demo purposes.
    ctx.scale = {"mm_per_px": 0.01, "px_per_mm": 100.0}
    return ctx
