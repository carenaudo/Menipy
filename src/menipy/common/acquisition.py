# src/menipy/common/acquisition.py
"""
Common acquisition utilities for loading images from files or cameras.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore


def _load_image(path: str) -> np.ndarray:
    """
    Attempt to load an image from disk using OpenCV when available, falling back
    to Pillow for environments without cv2.
    """
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img

    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            im = im.convert("L")
            return np.array(im)
    except Exception as exc:
        raise RuntimeError(f"Could not read image '{path}': {exc}") from exc


def from_file(paths: Sequence[str]) -> Sequence[np.ndarray]:
    """Load images from disk into grayscale numpy arrays."""
    frames = []
    for p in paths:
        frames.append(_load_image(str(p)))
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
