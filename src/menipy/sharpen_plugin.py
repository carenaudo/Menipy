from __future__ import annotations

import numpy as np
import cv2


def sharpen_filter(image: np.ndarray) -> np.ndarray:
    """Return a sharpened version of *image* using a simple kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


__all__ = ["sharpen_filter"]
