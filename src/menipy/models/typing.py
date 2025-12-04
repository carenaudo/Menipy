# src/menipy/models/typing.py
"""
Type aliases for numpy arrays used throughout Menipy.
"""
from __future__ import annotations

from typing import Union

import numpy.typing as npt
from pydantic_numpy.typing import (
    Np2DArrayUint8,
    Np3DArrayUint8,
    Np2DArrayFp64,
    Np1DArrayFp64,
)

# Image arrays: grayscale (H, W) or color (H, W, 3), uint8 typical for OpenCV.
ImageGray = Np2DArrayUint8
ImageBGR = Np3DArrayUint8
ImageAny = Union[ImageGray, ImageBGR]  # validated by shape checks
# Contours: N x 2 (x, y) in pixels or metric units depending on context
ContourArray = Np2DArrayFp64
# Time series: 1-D float arrays (seconds, value)
FloatVec = Np1DArrayFp64
