"""Frame model for representing images with metadata, calibration, and timing information."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic_numpy.typing import Np2DArrayUint8, Np3DArrayUint8

ImageGray: TypeAlias = Np2DArrayUint8
ImageBGR: TypeAlias = Np3DArrayUint8
ImageAny: TypeAlias = ImageGray | ImageBGR


class CameraMeta(BaseModel):
    """Minimal camera metadata recorded with frames."""

    device: str | None = Field(default=None, description="Camera name/id")
    fps: float | None = Field(default=None, gt=0)
    exposure_ms: float | None = Field(default=None, gt=0)
    resolution: tuple[int, int] | None = Field(
        default=None, description="(height, width)"
    )
    lens_mm: float | None = Field(default=None, gt=0)
    note: str | None = None


class Calibration(BaseModel):
    """Pixel-to-metric scaling and optional distortion parameters."""

    pixels_per_mm: float | None = Field(default=None, gt=0)
    mm_per_pixel: float | None = Field(default=None, gt=0)
    # If both provided, they are cross-checked for consistency
    k1: float | None = None  # radial distortion (optional)
    k2: float | None = None
    cx_cy: tuple[float, float] | None = Field(
        default=None, description="principal point (cx, cy) in pixels"
    )

    @field_validator("mm_per_pixel")
    @classmethod
    def _invertible_scale(cls, v, info):
        ppx = info.data.get("pixels_per_mm")
        if v and ppx and not np.isclose(v, 1.0 / ppx):
            raise ValueError("mm_per_pixel must equal 1 / pixels_per_mm")
        return v


class Frame(BaseModel):
    """
    A single image frame (silhouette or raw) plus timing and metadata.
    Accepts grayscale (H, W) or BGR (H, W, 3) uint8 arrays as used by OpenCV.
    """

    image: ImageAny = Field(description="np.uint8 image; shape (H,W) or (H,W,3)")
    timestamp: datetime | None = Field(default=None)
    ms_from_start: float | None = Field(default=None, ge=0)
    camera: CameraMeta | None = None
    calibration: Calibration | None = None

    @field_validator("image")
    @classmethod
    def _check_image_shape_dtype(cls, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError("image must be a numpy ndarray")
        if img.dtype != np.uint8:
            raise TypeError("image dtype must be uint8 (OpenCV-compatible)")
        if img.ndim == 2:
            # grayscale OK
            return img
        if img.ndim == 3 and img.shape[2] == 3:
            # color (assumed BGR in OpenCV)
            return img
        raise ValueError("image must have shape (H,W) or (H,W,3)")
