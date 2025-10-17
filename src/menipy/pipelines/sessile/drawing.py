from __future__ import annotations

from menipy.gui.overlay import draw_analysis_overlay
from .geometry import SessileMetrics


def draw_sessile_overlay(image, metrics: SessileMetrics):
    """Return a ``QPixmap`` with sessile-drop overlays."""
    return draw_analysis_overlay(image, metrics)


__all__ = ["draw_sessile_overlay"]