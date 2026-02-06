"""Geometric models for contours, geometry landmarks, and spatial features."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator
from .typing import ContourArray


class Point(BaseModel):
    """Represents a 2D point."""

    x: float
    y: float


class ROI(BaseModel):
    """Represents a rectangular region of interest."""

    x: int
    y: int
    width: int
    height: int


class Needle(BaseModel):
    """Represents the needle."""

    x: int
    y: int
    width: int
    height: int


class ContactLine(BaseModel):
    """Represents the contact line."""

    x1: float
    y1: float
    x2: float
    y2: float


class Contour(BaseModel):
    """
    Detected droplet (or meniscus) boundary.
    Coordinates are in pixels unless `units='mm'` and scaling applied.
    """

    xy: "ContourArray" = Field(description="array of shape (N, 2) with columns [x, y]")
    closed: bool = Field(default=True)
    units: Literal["px", "mm"] = Field(default="px")
    smoothing: Optional[float] = Field(default=None, ge=0, description="spline/fit λ")
    origin_hint: Optional[Tuple[float, float]] = Field(
        default=None, description="optional origin (x0, y0)"
    )

    @field_validator("xy")
    @classmethod
    def _check_xy(cls, arr: np.ndarray) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            raise TypeError("xy must be a numpy ndarray")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")
        if arr.dtype.kind not in ("f", "i"):
            raise TypeError("xy must be float or int array")
        return arr.astype(np.float64, copy=False)


class Geometry(BaseModel):
    """Geometric landmarks required by solvers."""

    apex_xy: Optional[Tuple[float, float]] = None  # pendant/sessile
    axis_x: Optional[float] = None  # symmetry axis x (px or mm)
    baseline_y: Optional[float] = None  # sessile: substrate y
    contact_region_px: Optional[Tuple[int, int]] = None  # index range around CL
    tilt_deg: float = Field(default=0.0)


class CaptiveBubbleGeometry(Geometry):
    """Geometry landmarks for captive bubble analysis."""

    ceiling_y: Optional[float] = Field(
        default=None, description="y-coordinate of chamber ceiling"
    )
    cap_depth_px: Optional[float] = Field(
        default=None, description="depth of bubble cap in pixels"
    )
