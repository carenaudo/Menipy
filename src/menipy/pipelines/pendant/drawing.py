"""
STUB: Pendant Pipeline - Drawing Stage

This file is a placeholder stub for the drawing and visualization utilities.

TODO: Implement drawing stage for pendant pipeline
      - Define stage-specific logic
      - Add proper error handling
      - Write unit tests
      - Update documentation

See pendant_plan_pipeline.md for implementation details.
"""

from __future__ import annotations

from menipy.gui.overlay import draw_analysis_overlay
from .geometry import PendantMetrics


def draw_pendant_overlay(image, metrics: PendantMetrics):
    """Return a ``QPixmap`` with pendant-drop overlays."""
    return draw_analysis_overlay(image, metrics)


__all__ = ["draw_pendant_overlay"]
