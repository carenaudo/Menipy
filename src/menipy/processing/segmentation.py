"""Segmentation routines for droplet images."""

import cv2
import numpy as np
from skimage import filters, morphology


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Segment image using Otsu's global threshold."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    thresh_val, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def adaptive_threshold(image: np.ndarray, block_size: int = 51, offset: int = 10) -> np.ndarray:
    """Segment image using adaptive mean thresholding."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, block_size, offset)
    return mask


def morphological_cleanup(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Clean up a binary mask using morphological opening and closing."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def external_contour_mask(mask: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest external contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    ext = np.zeros_like(mask)
    cv2.drawContours(ext, [largest], -1, 255, -1)
    return ext


def find_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Return external contours with sub-pixel precision."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result: list[np.ndarray] = []
    for c in contours:
        if c.size > 0:
            result.append(c.squeeze(1).astype(np.float32))
    return result


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    """Return the largest external contour as floats or ``None`` if none found."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest.squeeze(1).astype(np.float32)


def ml_segment(image: np.ndarray) -> np.ndarray:
    """Placeholder ML-based segmentation.

    This stub mimics an ML model by applying Otsu thresholding followed by
    morphological cleanup. It allows the GUI to toggle an "ML" option without
    requiring heavy dependencies.
    """
    mask = otsu_threshold(image)
    return morphological_cleanup(mask, kernel_size=3, iterations=1)

