from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from menipy.analysis import extract_external_contour, compute_pendant_metrics


@dataclass
class HelperBundle:
    px_per_mm: float
    needle_diam_mm: float | None = None


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
    metrics = compute_pendant_metrics(
        contour.astype(float),
        px_per_mm=helpers.px_per_mm,
        needle_diam_mm=helpers.needle_diam_mm,
    )
    return PendantMetrics(
        contour=contour,
        apex=metrics["apex"],
        diameter_line=metrics["diameter_line"],
        contact_line=metrics.get("contact_line"),
        diameter_center=metrics.get("diameter_center"),
        derived=metrics,
    )


__all__ = ["analyze", "PendantMetrics", "HelperBundle"]
