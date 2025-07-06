"""Pendant drop analysis functions."""

from __future__ import annotations

import numpy as np

from .commons import compute_drop_metrics


def compute_metrics(
    contour: np.ndarray, px_per_mm: float, needle_diam_mm: float | None = None
) -> dict:
    """Return pendant-drop metrics for ``contour``."""
    return compute_drop_metrics(contour, px_per_mm, "pendant", needle_diam_mm)

__all__ = ["compute_metrics"]
