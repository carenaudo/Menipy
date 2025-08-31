# src/adsa/common/geometry.py
from __future__ import annotations
import numpy as np

def find_apex(xy: np.ndarray, prefer_above: float | None = None) -> tuple[float, float]:
    """Apex = min y (image y-down)."""
    i = np.argmin(xy[:, 1])
    return float(xy[i, 0]), float(xy[i, 1])

def find_symmetry_axis(xy: np.ndarray) -> float:
    """Approximate symmetry axis as mean x."""
    return float(np.mean(xy[:, 0]))

def find_baseline(xy: np.ndarray) -> float:
    """Rough baseline: max y among the lowest 10% points."""
    q = np.quantile(xy[:, 1], 0.9)
    return float(q)

def run(ctx):
    """Generic geometry pass populating apex & axis; pipelines may override."""
    xy = ctx.contour.xy
    ctx.geometry = {
        "apex_xy": find_apex(xy),
        "axis_x": find_symmetry_axis(xy),
    }
    return ctx
