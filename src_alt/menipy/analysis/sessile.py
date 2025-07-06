"""Sessile drop analysis functions."""

from __future__ import annotations

import numpy as np

from .commons import compute_drop_metrics


def compute_metrics(
    contour: np.ndarray,
    px_per_mm: float,
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> dict:
    """Return sessile-drop metrics for ``contour``."""
    return compute_drop_metrics(
        contour, px_per_mm, "contact-angle", substrate_line=substrate_line
    )

__all__ = ["compute_metrics"]
