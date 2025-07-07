from __future__ import annotations

import numpy as np
from PySide6.QtGui import QPixmap

# Import the overlay helper directly to avoid a circular import with the
# ``gui`` package's ``__init__`` when this module is imported from within that
# package.
from ...gui.overlay import draw_drop_overlay
from .geometry import SessileMetrics


def _axis_line(metrics: SessileMetrics) -> tuple[tuple[int, int], tuple[int, int]]:
    line_dir = np.subtract(metrics.substrate_line[1], metrics.substrate_line[0])
    p1 = np.array(metrics.diameter_line[0], float)
    apex_pt = np.array(metrics.apex, float)
    t = float(np.dot(apex_pt - p1, line_dir)) / float(np.dot(line_dir, line_dir))
    t = np.clip(t, 0.0, 1.0)
    foot_pt = p1 + t * line_dir
    return (
        tuple(np.round(foot_pt).astype(int)),
        tuple(np.round(apex_pt).astype(int)),
    )


def draw_overlays(image: np.ndarray, metrics: SessileMetrics) -> QPixmap:
    """Return a ``QPixmap`` with sessile-drop overlays."""
    axis_line = _axis_line(metrics)
    contact_line = (metrics.p1, metrics.p2)
    return draw_drop_overlay(
        image,
        metrics.contour,
        diameter_line=metrics.diameter_line,
        axis_line=axis_line,
        contact_line=contact_line,
        apex=metrics.apex,
        contact_pts=contact_line,
    )


__all__ = ["draw_overlays"]
