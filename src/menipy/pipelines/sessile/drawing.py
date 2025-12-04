"""
STUB: Sessile Pipeline - Drawing Stage

This file is a placeholder stub for the drawing and visualization utilities.

TODO: Implement drawing stage for sessile pipeline
      - Define stage-specific logic
      - Add proper error handling
      - Write unit tests
      - Update documentation

See sessile_plan_pipeline.md for implementation details.
"""

from __future__ import annotations

from menipy.gui.overlay import draw_analysis_overlay
from .geometry import SessileMetrics


def draw_sessile_overlay(image, metrics: SessileMetrics):
    """Return a ``QPixmap`` with sessile-drop overlays."""
    return draw_analysis_overlay(image, metrics)


__all__ = ["draw_sessile_overlay"]