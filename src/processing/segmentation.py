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

