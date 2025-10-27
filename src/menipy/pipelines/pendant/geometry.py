from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from menipy.common.edge_detection import extract_external_contour
from menipy.common.metrics import find_apex_index
from .metrics import compute_pendant_metrics


@dataclass
class HelperBundle:
    px_per_mm: float
    needle_diam_mm: float | None = None
    delta_rho: float = 998.8
    g: float = 9.80665
    apex_window_px: int = 10


@dataclass
class PendantMetrics:
    contour: np.ndarray
    apex: tuple[int, int]
    diameter_line: tuple[tuple[int, int], tuple[int, int]]
    contact_line: tuple[tuple[int, int], tuple[int, int]] | None
    diameter_center: tuple[int, int] | None
    derived: dict[str, float]


def analyze(frame: np.ndarray, helpers: HelperBundle) -> PendantMetrics:
    """Return pendant-drop metrics and geometry from ``frame``."""
    contour = extract_external_contour(frame)
    apex_idx = find_apex_index(contour, "pendant")
    apex = tuple(contour[apex_idx].astype(int))
    metrics = compute_pendant_metrics(
        contour.astype(float),
        px_per_mm=helpers.px_per_mm,
        needle_diam_mm=helpers.needle_diam_mm,
        apex=apex,
        delta_rho=helpers.delta_rho,
        g=helpers.g,
        apex_window_px=helpers.apex_window_px,
    )
    return PendantMetrics(
        contour=contour,
        apex=apex,
        diameter_line=metrics["diameter_line"],
        contact_line=metrics.get("contact_line"),
        diameter_center=metrics.get("diameter_center"),
        derived=metrics,
    )


class PendantPipeline:
    """A pipeline for analyzing pendant drops."""

    name = "pendant"


__all__ = ["analyze", "PendantMetrics", "HelperBundle", "PendantPipeline"]
