"""Common metrics calculations for droplet analysis."""

import numpy as np

from menipy.math.apex import detect_apex


def compute_drop_metrics(
    contour: np.ndarray,
    px_per_mm: float,
    mode: str,
    needle_diam_mm: float | None = None,
) -> dict:
    """
    Compute geometric metrics for a detected drop contour.

    Args:
        contour: (N, 2) array of contour points.
        px_per_mm: Scale factor.
        mode: Analysis mode ('sessile' or 'pendant').
        needle_diam_mm: Optional needle diameter for calibration verification.

    Returns:
        Dictionary containing calculated metrics (volume, dimensions, etc.).
        Currently returns a placeholder structure.
    """
    return {"apex": (0, 0), "diameter_mm": 0.0, "height_mm": 0.0, "volume_uL": 0.0}


def find_apex_index(contour: np.ndarray, mode: str) -> int:
    """Find the index of the apex point in the contour.

    Args:
        contour: (N, 2) array of contour points.
        mode: 'sessile' (top), 'pendant' (bottom), or 'captive_bubble'.

    Returns:
        Index of the contour vertex closest to the detected apex.
    """
    pts = np.asarray(contour, dtype=float)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    if pts.shape[0] == 0:
        return 0

    try:
        res = detect_apex(pts, mode=mode, refine=False)
        target = np.array(res.point, dtype=float)
        dists = np.sum((pts - target) ** 2, axis=1)
        return int(np.argmin(dists))
    except Exception:
        if mode.lower() in ("pendant", "captive_bubble"):
            return int(np.argmax(pts[:, 1]))
        return int(np.argmin(pts[:, 1]))

