"""
Shared image processing utilities.
"""
from __future__ import annotations
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Ensure image is grayscale."""
    if img.ndim == 2:
        return img
    
    # Try using cv2 first
    if cv2 is not None and img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Fallback to simple luminance
    if img.ndim == 3:
        return (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]).astype(
            np.uint8
        )
    return img


def edges_to_xy(
    edges: np.ndarray,
    min_len: int = 0,
    max_len: int = 100000,
) -> np.ndarray:
    """Convert a binary edges mask to an (N,2) contour array (largest external contour)."""
    if edges is None:
        return np.empty((0, 2), float)
    
    if cv2 is None:
        # Simple fallback if no OpenCV
        ys, xs = np.nonzero(edges)
        if xs.size == 0:
            return np.empty((0, 2), float)
        xy = np.column_stack([xs, ys]).astype(float)
        return xy
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), float)
    
    # Filter by length
    valid_cnts = [c for c in contours if min_len <= len(c) <= max_len]
    if not valid_cnts:
        return np.empty((0, 2), float)
    
    # Select largest by area
    c = max(valid_cnts, key=cv2.contourArea)
    xy = c.reshape(-1, 2).astype(float)
    return xy
