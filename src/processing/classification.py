import numpy as np
from typing import Literal

from .detection import Droplet
from ..utils import get_calibration


def classify_drop_mode(droplet: Droplet) -> Literal["pendant", "sessile", "unknown"]:
    """Classify whether a drop is pendant or sessile.

    Parameters
    ----------
    droplet:
        Droplet geometry obtained from detection.

    Returns
    -------
    Literal["pendant", "sessile", "unknown"]
        Detected drop mode or "unknown" if confidence is low.
    """
    if droplet.contact_px is None:
        return "unknown"

    try:
        x1, y1, x2, y2 = droplet.contact_px  # type: ignore[misc]
    except Exception:
        # contact line not provided as segment
        return "unknown"

    dx = float(x2 - x1)
    dy = float(y2 - y1)
    width_px = np.hypot(dx, dy)
    if width_px < 20.0:
        return "unknown"

    # Image coordinates use Y going downward. Flip the conventional
    # surface-normal direction so ``n`` points out of the solid.
    n = np.array([dy, -dx], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm == 0.0:
        return "unknown"
    n /= norm

    mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)
    apex_vec = np.array(droplet.apex_px, dtype=float) - mid
    sign = float(np.dot(apex_vec, n))

    px_to_mm = 1.0 / get_calibration().pixels_per_mm
    gap_mm = abs(sign) * px_to_mm

    if gap_mm < 0.05:
        return "unknown"
    elif sign < 0:
        return "pendant"
    else:
        return "sessile"
