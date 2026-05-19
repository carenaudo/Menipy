"""Common, low-level geometric utilities."""

from __future__ import annotations

import cv2
import numpy as np


def extract_external_contour(frame: np.ndarray) -> np.ndarray:
    """Extracts the largest external contour from a binary image.

    Args:
        frame (np.ndarray): The input binary image (mask).

    Returns:
        np.ndarray: The largest external contour found.
    """
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return contour


__all__ = ["extract_external_contour"]
