from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from ...analysis import extract_external_contour, compute_sessile_metrics


@dataclass
class HelperBundle:
    px_per_mm: float


@dataclass
class SessileMetrics:
    contour: np.ndarray
    apex: tuple[int, int]
    diameter_line: tuple[tuple[int, int], tuple[int, int]]
    p1: tuple[int, int]
    p2: tuple[int, int]
    substrate_line: tuple[tuple[int, int], tuple[int, int]]
    derived: dict[str, float]


def analyze(
    frame: np.ndarray,
    helpers: HelperBundle,
    substrate: tuple[tuple[int, int], tuple[int, int]],
    drop_side: Literal["left", "right", "auto"] = "auto",
) -> SessileMetrics:
    """Return sessile-drop metrics and geometry from ``frame``."""
    contour = extract_external_contour(frame)
    metrics = compute_sessile_metrics(
        contour.astype(float), helpers.px_per_mm, substrate_line=substrate
    )
    p1, p2 = metrics["contact_line"] if metrics.get("contact_line") else ((0, 0), (0, 0))
    return SessileMetrics(
        contour=contour,
        apex=metrics["apex"],
        diameter_line=metrics["diameter_line"],
        p1=p1,
        p2=p2,
        substrate_line=substrate,
        derived=metrics,
    )


__all__ = ["analyze", "SessileMetrics", "HelperBundle"]
