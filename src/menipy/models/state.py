"""Runtime state models for Menipy (migrated from datatypes.py).

This module contains mutable/runtime models such as PreprocessingState
and small helper constructors. These types are intended to replace the
ones previously living in `datatypes.py`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .frame import Calibration, CameraMeta, Frame
from .geometry import Contour


class MarkerSet(BaseModel):
    """Interactive markers collected from preview interactions."""

    drop_center: tuple[float, float] | None = None
    contact_line_anchors: list[tuple[float, float]] = Field(default_factory=list)
    background_samples: list[tuple[float, float]] = Field(default_factory=list)

    def add_anchor(self, point: tuple[float, float]) -> None:
        self.contact_line_anchors.append(point)

    def clear(self) -> None:
        self.drop_center = None
        self.contact_line_anchors.clear()
        self.background_samples.clear()


class PreprocessingStageRecord(BaseModel):
    """Audit record describing a single stage execution."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class PreprocessingState(BaseModel):
    """Mutable buffers shared across preprocessing helpers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    roi_bounds: tuple[int, int, int, int] | None = None
    roi_mask: np.ndarray | None = None
    raw_roi: np.ndarray | None = None
    working_roi: np.ndarray | None = None
    filtered_roi: np.ndarray | None = None
    normalized_roi: np.ndarray | None = None
    scale: tuple[float, float] = Field(default=(1.0, 1.0))
    contact_line_mask: np.ndarray | None = None
    contact_line_presence: bool = False
    markers: MarkerSet = Field(default_factory=MarkerSet)
    history: list[PreprocessingStageRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def clone(self) -> PreprocessingState:
        """clone.

        Returns
        -------
        type
        Description.
        """
        return self.model_copy(deep=True)

        # Convenience constructors (migrated)


def make_frame(
    image: np.ndarray,
    timestamp: datetime | None = None,
    ms_from_start: float | None = None,
    camera: CameraMeta | None = None,
    calibration: Calibration | None = None,
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
    smoothing: float | None = None,
    origin_hint: tuple[float, float] | None = None,
) -> Contour:
    """Helper to create a validated Contour."""
    return Contour(
        xy=xy, closed=closed, units=units, smoothing=smoothing, origin_hint=origin_hint
    )
