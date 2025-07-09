from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from ...analysis import (
    extract_external_contour,
    compute_sessile_metrics_alt,
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
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None = None,
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

    if contact_points is not None:
        from ...physics.contact_geom import (
            contour_line_intersection_near,
            line_params,
        )

        (m1, m2) = contact_points
        a, b, c = line_params(substrate[0], substrate[1])
        try:
            left_pt, _ = contour_line_intersection_near(
                clean_contour.astype(float), a, b, c, m1
            )
            right_pt, _ = contour_line_intersection_near(
                clean_contour.astype(float), a, b, c, m2
            )
            p1_fit, p2_fit = tuple(left_pt), tuple(right_pt)
        except Exception:
            pass

    metrics = compute_sessile_metrics_alt(
        clean_contour.astype(float), helpers.px_per_mm, substrate_line=substrate
    )
    p1, p2 = metrics.get("contact_line", (p1_fit, p2_fit))
    metrics["contact_line"] = (p1, p2)
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
