"""Explicit spatial-calibration helpers independent of needle detection."""

from __future__ import annotations

import math

import numpy as np


def px_per_mm_from_known_distance(points, distance_mm: float) -> float:
    """Return an explicit scale from a two-point reference segment.

    The segment is measured in image pixels and may be tilted. Callers own the
    UI/provenance; invalid input raises rather than manufacturing a scale.
    """
    segment = np.asarray(points, dtype=float).reshape(2, 2)
    distance = float(distance_mm)
    length_px = float(np.linalg.norm(segment[1] - segment[0]))
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("Known reference distance must be finite and positive.")
    if not math.isfinite(length_px) or length_px <= 0:
        raise ValueError("Reference segment must have nonzero pixel length.")
    return length_px / distance
