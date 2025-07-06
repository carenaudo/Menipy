"""Preprocessing helpers ported from legacy code."""

from __future__ import annotations

import numpy as np
import cv2


def clean_frame(frame: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Return a cleaned binary mask using morphological opening and closing."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed
