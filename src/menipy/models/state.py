"""Runtime state models for Menipy (migrated from datatypes.py).

This module contains mutable/runtime models such as PreprocessingState
and small helper constructors. These types are intended to replace the
ones previously living in `datatypes.py`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from .frame import Frame, Calibration, CameraMeta
from .geometry import Contour


class MarkerSet(BaseModel):
    """Interactive markers collected from preview interactions."""

    drop_center: Optional[Tuple[float, float]] = None
    contact_line_anchors: List[Tuple[float, float]] = Field(default_factory=list)
    background_samples: List[Tuple[float, float]] = Field(default_factory=list)

    def add_anchor(self, point: Tuple[float, float]) -> None:
        self.contact_line_anchors.append(point)

    def clear(self) -> None:
        self.drop_center = None
        self.contact_line_anchors.clear()
        self.background_samples.clear()


class PreprocessingStageRecord(BaseModel):
    """Audit record describing a single stage execution."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PreprocessingState(BaseModel):
    """Mutable buffers shared across preprocessing helpers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    roi_bounds: Optional[Tuple[int, int, int, int]] = None
    roi_mask: Optional[np.ndarray] = None
    raw_roi: Optional[np.ndarray] = None
    working_roi: Optional[np.ndarray] = None
    filtered_roi: Optional[np.ndarray] = None
    normalized_roi: Optional[np.ndarray] = None
    scale: Tuple[float, float] = Field(default=(1.0, 1.0))
    contact_line_mask: Optional[np.ndarray] = None
    contact_line_presence: bool = False
    markers: MarkerSet = Field(default_factory=MarkerSet)
    history: List[PreprocessingStageRecord] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def clone(self) -> "PreprocessingState":
        return self.model_copy(deep=True)


# Convenience constructors (migrated)
def make_frame(
    image: np.ndarray,
    timestamp: Optional[datetime] = None,
    ms_from_start: Optional[float] = None,
    camera: Optional[CameraMeta] = None,
    calibration: Optional[Calibration] = None,
) -> Frame:
    """Helper to create a validated Frame instance."""
    return Frame(
        image=image,
        timestamp=timestamp,
        ms_from_start=ms_from_start,
        camera=camera,
        calibration=calibration,
    )


def make_contour(
    xy: np.ndarray,
    closed: bool = True,
    units: Literal["px", "mm"] = "px",
    smoothing: Optional[float] = None,
    origin_hint: Optional[Tuple[float, float]] = None,
) -> Contour:
    """Helper to create a validated Contour."""
    return Contour(
        xy=xy, closed=closed, units=units, smoothing=smoothing, origin_hint=origin_hint
    )
