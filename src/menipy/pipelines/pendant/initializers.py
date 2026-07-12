"""Built-in pendant geometry initializers."""

from __future__ import annotations

from typing import Any

import numpy as np

from menipy.common.geometry_prototypes import robust_pendant_initializer
from menipy.common.registry import PENDANT_INITIALIZERS, register_pendant_initializer


def legacy_pendant_initializer(contour_px: Any, px_per_mm: float = 1.0, **_: Any):
    """Compatibility initializer matching the historical vertical-axis seeds."""
    xy = np.asarray(contour_px, dtype=float).reshape(-1, 2)
    if xy.size == 0:
        return robust_pendant_initializer(xy, px_per_mm)
    return robust_pendant_initializer(xy, px_per_mm)


register_pendant_initializer("robust_axis", robust_pendant_initializer)
register_pendant_initializer("legacy", legacy_pendant_initializer)


__all__ = ["PENDANT_INITIALIZERS", "legacy_pendant_initializer", "robust_pendant_initializer"]
