# src/menipy/models/datatypes.py
"""Common data types and structures for analysis records and preprocessing state."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    # PhysicsParams is not exported from models.physics; use a loose alias for typing
    from typing import Any as PhysicsParams

    from .frame import Calibration, CameraMeta, Frame
    from .geometry import Contour, Geometry

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .result import CapillaryRiseFit, OscillationFit, YoungLaplaceFit


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

        # ---- Aggregated analysis record --------------------------------------------


class AnalysisRecord(BaseModel):
    """
    One complete run of a pipeline stage sequence.
    Stores the minimal artifacts needed for reproducibility.
    """

    kind: Literal["pendant", "sessile", "oscillating", "capillary_rise"]
    frame: Frame | None = None  # single image or representative frame
    contour: Contour | None = None
    geometry: Geometry | None = None
    calibration: Calibration | None = None
    physics: PhysicsParams | None = None

    fit_young_laplace: YoungLaplaceFit | None = None
    fit_oscillation: OscillationFit | None = None
    fit_capillary: CapillaryRiseFit | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0")

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, k: str) -> str:
        return k


# ---- Convenience constructors ----------------------------------------------


def make_frame(
    image: np.ndarray,
    timestamp: datetime | None = None,
    ms_from_start: float | None = None,
    camera: CameraMeta | None = None,
    calibration: Calibration | None = None,
) -> Frame:
    """Helper to create a validated Frame."""
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
