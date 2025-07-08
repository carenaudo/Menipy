from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from ...analysis import (
    extract_external_contour,
    compute_sessile_metrics,
    smooth_contour_segment,
)


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
    try:
        clean_contour, p1_fit, p2_fit = smooth_contour_segment(
            contour.astype(float), substrate, drop_side if drop_side != "auto" else "left"
        )
    except Exception:
        clean_contour = contour.astype(float)
        p1_fit, p2_fit = (0.0, 0.0), (0.0, 0.0)

    metrics = compute_sessile_metrics(
        clean_contour.astype(float), helpers.px_per_mm, substrate_line=substrate
    )
    p1, p2 = metrics.get("contact_line", (p1_fit, p2_fit))
    return SessileMetrics(
        contour=clean_contour.astype(float),
        apex=metrics["apex"],
        diameter_line=metrics["diameter_line"],
        p1=p1,
        p2=p2,
        substrate_line=substrate,
        derived=metrics,
    )


__all__ = ["analyze", "SessileMetrics", "HelperBundle"]
