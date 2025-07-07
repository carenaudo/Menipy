from __future__ import annotations

import numpy as np
from PySide6.QtGui import QPixmap

from ...gui import draw_drop_overlay
from .geometry import PendantMetrics


def draw_overlays(image: np.ndarray, metrics: PendantMetrics) -> QPixmap:
    """Return a ``QPixmap`` with pendant-drop overlays."""
    center_pt = metrics.diameter_center
    contact_line = metrics.contact_line
    center_apex_line = None
    center_contact_line = None
    if center_pt is not None:
        center_apex_line = (center_pt, metrics.apex)
        if contact_line is not None:
            cl_center = (
                (contact_line[0][0] + contact_line[1][0]) // 2,
                (contact_line[0][1] + contact_line[1][1]) // 2,
            )
            center_contact_line = (center_pt, cl_center)
    return draw_drop_overlay(
        image,
        metrics.contour,
        diameter_line=metrics.diameter_line,
        contact_line=contact_line,
        apex=metrics.apex,
        contact_pts=contact_line,
        center_pt=center_pt,
        center_apex_line=center_apex_line,
        center_contact_line=center_contact_line,
    )


__all__ = ["draw_overlays"]
