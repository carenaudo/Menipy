"""Droplet metric helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..models.properties import droplet_volume
from .region import close_droplet, _signed_distance


def _apex_point(
    contour: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray, mode: str
) -> np.ndarray:
    """Return the apex relative to ``line_pt``/``line_dir``.

    The apex is taken as the contour point with the largest (sessile) or
    smallest (pendant) signed distance to the substrate line.  If several
    points share that extreme distance, the candidate whose projection onto
    the substrate line is in the middle of the group is returned.  This
    avoids biasing the apex toward either side when a flat region occurs.
    """

    dist = _signed_distance(contour, line_pt, line_dir)
    if mode == "sessile":
        extreme = dist.max()
        idxs = np.where(np.isclose(dist, extreme))[0]
    else:
        extreme = dist.min()
        idxs = np.where(np.isclose(dist, extreme))[0]

    pts = contour[idxs]
    if len(idxs) == 1:
        return pts[0]

    # choose the candidate centred along the substrate direction
    t = (pts - line_pt) @ line_dir
    order = np.argsort(t)
    mid = len(order) // 2
    return pts[order[mid]]


def metrics_sessile(
    contour: np.ndarray,
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    px_per_mm: float,
) -> dict:
    """Return sessile droplet metrics relative to a substrate line."""
    p1, p2 = line
    mask, c1, c2 = close_droplet(contour, p1, np.subtract(p2, p1), "sessile")
    apex = _apex_point(contour, np.array(p1, float), np.subtract(p2, p1), "sessile")
    nvec = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]], float)
    nvec /= np.hypot(nvec[0], nvec[1])
    h_px = abs((apex - p1) @ nvec)
    w_px = np.linalg.norm(c2 - c1)
    volume = droplet_volume(mask, 1.0 / px_per_mm)
    return {
        "diameter_mm": float(w_px / px_per_mm),
        "rb_mm": float(w_px / (2.0 * px_per_mm)),
        "apex": (int(round(apex[0])), int(round(apex[1]))),
        "height_mm": float(h_px / px_per_mm),
        "volume_mm3": float(volume) if volume is not None else None,
        "mask": mask,
        "p1": (float(c1[0]), float(c1[1])),
        "p2": (float(c2[0]), float(c2[1])),
    }


def metrics_pendant(
    contour: np.ndarray,
    line: Tuple[Tuple[float, float], Tuple[float, float]],
    px_per_mm: float,
) -> dict:
    """Return pendant droplet metrics relative to a needle line."""
    p1, p2 = line
    mask, c1, c2 = close_droplet(contour, p1, np.subtract(p2, p1), "pendant")
    apex = _apex_point(contour, np.array(p1, float), np.subtract(p2, p1), "pendant")
    nvec = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]], float)
    nvec /= np.hypot(nvec[0], nvec[1])
    h_px = abs((apex - p1) @ nvec)
    w_px = np.linalg.norm(c2 - c1)
    volume = droplet_volume(mask, 1.0 / px_per_mm)
    return {
        "diameter_mm": float(w_px / px_per_mm),
        "rb_mm": float(w_px / (2.0 * px_per_mm)),
        "apex": (int(round(apex[0])), int(round(apex[1]))),
        "height_mm": float(h_px / px_per_mm),
        "volume_mm3": float(volume) if volume is not None else None,
        "mask": mask,
        "p1": (float(c1[0]), float(c1[1])),
        "p2": (float(c2[0]), float(c2[1])),
    }

