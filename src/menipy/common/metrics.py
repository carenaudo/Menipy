"""
Common metrics calculations for droplet analysis.
"""

import numpy as np


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
    """
    Find the index of the apex point in the contour.

    Args:
        contour: (N, 2) array of contour points.
        mode: 'sessile' (top) or 'pendant' (bottom).

    Returns:
        Index of the apex point. Defaults to 0 for placeholder.
    """
    return 0
