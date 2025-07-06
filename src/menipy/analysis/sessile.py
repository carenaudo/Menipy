"""Sessile drop analysis functions."""

from __future__ import annotations

import numpy as np

from .commons import compute_drop_metrics, find_apex_index
from ..physics.contact_geom import geom_metrics


def compute_metrics(
    contour: np.ndarray,
    px_per_mm: float,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> dict:
    """Return sessile-drop metrics for ``contour``."""
    if substrate_line is None:
        return compute_drop_metrics(contour, px_per_mm, "contact-angle")

    apex_idx = find_apex_index(contour, "contact-angle")
    geo = geom_metrics(substrate_line[0], substrate_line[1], contour, apex_idx, px_per_mm)
    droplet_poly = geo.pop("droplet_poly")
    metrics = compute_drop_metrics(
        droplet_poly.astype(float),
        px_per_mm,
        "contact-angle",
        substrate_line=substrate_line,
    )
    metrics.update(geo)
    return metrics

__all__ = ["compute_metrics"]
