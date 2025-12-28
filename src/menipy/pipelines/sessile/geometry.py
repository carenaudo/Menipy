from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from menipy.common.edge_detection import extract_external_contour
from menipy.common.metrics import find_apex_index
from .metrics import compute_sessile_metrics


@dataclass
class HelperBundle:
    px_per_mm: float
    substrate_line: tuple[tuple[int, int], tuple[int, int]] | None = None
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None = None
    delta_rho: float = 998.8
    g: float = 9.80665
    contact_point_tolerance_px: float = 20.0


@dataclass
class SessileMetrics:
    contour: np.ndarray
    apex: tuple[int, int]
    diameter_line: tuple[tuple[int, int], tuple[int, int]]
    contact_line: tuple[tuple[int, int], tuple[int, int]] | None
    diameter_center: tuple[int, int] | None
    derived: dict[str, float]


def analyze(frame: np.ndarray, helpers: HelperBundle) -> SessileMetrics:
    """Return sessile-drop metrics and geometry from ``frame``."""
    contour = extract_external_contour(frame)
    apex_idx = find_apex_index(contour, "sessile")
    apex = tuple(contour[apex_idx].astype(int))
    metrics = compute_sessile_metrics(
        contour.astype(float),
        px_per_mm=helpers.px_per_mm,
        substrate_line=helpers.substrate_line,
        apex=apex,
        contact_point_tolerance_px=helpers.contact_point_tolerance_px,
    )
    return SessileMetrics(
        contour=contour,
        apex=apex,
        diameter_line=metrics["diameter_line"],
        contact_line=metrics.get("contact_line"),
        diameter_center=metrics.get("diameter_center"),
        derived=metrics,
    )


class SessilePipeline:
    """A pipeline for analyzing sessile drops."""

    name = "sessile"


__all__ = ["analyze", "SessileMetrics", "HelperBundle", "SessilePipeline"]
