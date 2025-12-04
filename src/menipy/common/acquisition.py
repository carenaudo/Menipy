# src/menipy/common/acquisition.py
"""
Common acquisition utilities for loading images from files or cameras.
"""
"""
Edge detection utilities and pipeline stage logic.
"""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

def from_file(paths: Sequence[str]) -> Sequence[np.ndarray]:
    """Load images from disk (placeholder). TODO: implement with cv2.imread."""
    import cv2  # uncomment when using OpenCV
    frames = []
    for p in paths:
         img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
         assert img is not None, f"Could not read {p}"
         frames.append(img)
        # frames.append(np.zeros((480, 640), dtype=np.uint8))  # placeholder
    return frames

def from_camera(device: int = 0, n_frames: int = 1) -> Sequence[np.ndarray]:
    """Grab frames from a camera (placeholder)."""
    # TODO: implement cv2.VideoCapture(device)
    return [np.zeros((480, 640), dtype=np.uint8) for _ in range(n_frames)]

def from_camera_or_file(expected: str = "silhouette", **kwargs):
    """Convenience entry used by pipelines."""
    if "paths" in kwargs:
        return from_file(kwargs["paths"])
    return from_camera(**kwargs)