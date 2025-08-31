"""Droplet region utilities."""

from __future__ import annotations

from typing import Literal, Tuple

import cv2
import numpy as np


_EPS = 2.0


def _signed_distance(points: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray) -> np.ndarray:
    """Return signed distances of ``points`` to the line defined by ``line_pt`` and ``line_dir``."""
    nvec = np.array([-line_dir[1], line_dir[0]], float)
    norm = np.hypot(nvec[0], nvec[1])
    if norm == 0:
        raise ValueError("line_dir cannot be zero")
    nvec /= norm
    return (points - line_pt) @ nvec


def _cluster_points(points: np.ndarray, eps: float = 3.0) -> list[np.ndarray]:
    """Cluster ``points`` using a simple distance-based BFS."""
    remaining = list(range(len(points)))
    clusters = []
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            for idx in list(remaining):
                if np.linalg.norm(points[idx] - points[seed]) <= eps:
                    remaining.remove(idx)
                    cluster.append(idx)
                    changed = True
        clusters.append(points[cluster])
    return clusters


def close_droplet(
    contour: np.ndarray,
    line_pt: Tuple[float, float],
    line_dir: Tuple[float, float],
    mode: Literal["sessile", "pendant"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a closed droplet mask and contact points ``p1``, ``p2``."""
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must have shape (N,2)")
    if mode not in {"sessile", "pendant"}:
        raise ValueError("mode must be 'sessile' or 'pendant'")

    contour = contour.astype(float)
    line_pt = np.asarray(line_pt, float)
    line_dir = np.asarray(line_dir, float)

    dist = _signed_distance(contour, line_pt, line_dir)
    keep = dist > _EPS if mode == "sessile" else dist < -_EPS
    kept = contour[keep]
    if kept.size == 0:
        kept = contour

    # --- locate intersections -------------------------------------------------
    inter: list[np.ndarray] = []
    dist_next = np.roll(dist, -1)
    for p, q, dp, dq in zip(contour, np.roll(contour, -1, axis=0), dist, dist_next):
        if dp == dq:
            continue
        if dp * dq <= 0:
            t = dp / (dp - dq)
            inter.append(p + t * (q - p))
    if len(inter) < 2:
        raise ValueError("line does not intersect contour")
    inter_pts = np.array(inter)

    if len(inter_pts) > 2:
        clusters = _cluster_points(inter_pts, eps=3.0)
        centroids = np.array([c.mean(axis=0) for c in clusters])
    else:
        centroids = inter_pts

    tvals = centroids @ line_dir
    if len(centroids) > 2:
        diff = tvals[:, None] - tvals[None, :]
        i, j = divmod(np.abs(diff).argmax(), diff.shape[0])
        p1, p2 = centroids[min(i, j)], centroids[max(i, j)]
    else:
        order = np.argsort(tvals)
        p1, p2 = centroids[order[0]], centroids[order[1]]

    # --- mask closing ---------------------------------------------------------
    x_min = int(np.floor(min(kept[:, 0].min(), p1[0], p2[0])))
    x_max = int(np.ceil(max(kept[:, 0].max(), p1[0], p2[0])))
    y_min = int(np.floor(min(kept[:, 1].min(), p1[1], p2[1])))
    y_max = int(np.ceil(max(kept[:, 1].max(), p1[1], p2[1])))
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = np.round(kept - [x_min, y_min]).astype(np.int32)
    if shifted.size > 0:
        cv2.fillPoly(mask, [shifted], 255)
    cv2.line(
        mask,
        tuple(np.round(p1 - [x_min, y_min]).astype(int)),
        tuple(np.round(p2 - [x_min, y_min]).astype(int)),
        255,
        1,
    )

    return mask, p1, p2

