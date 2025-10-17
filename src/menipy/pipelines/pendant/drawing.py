from __future__ import annotations

from menipy.gui.overlay import draw_analysis_overlay
from .geometry import PendantMetrics


def draw_pendant_overlay(image, metrics: PendantMetrics):
    """Return a ``QPixmap`` with pendant-drop overlays."""
    return draw_analysis_overlay(image, metrics)


__all__ = ["draw_pendant_overlay"]
