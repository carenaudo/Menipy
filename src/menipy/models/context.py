"""Context model for sharing state between pipeline stages."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .config import EdgeDetectionSettings, PreprocessingSettings
from .fit import Fit
from .frame import Frame
from .geometry import Contour, Geometry
from .result import Result
from .state import MarkerSet


class Context(BaseModel):
    """
    Mutable bag of state shared across pipeline stages.
    Pipelines freely attach fields here. Commonly used keys are predeclared.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Acquisition / images
    frames: np.ndarray | list[np.ndarray] | list[Frame] | None = (
        None  # np.ndarray or list[np.ndarray]
    )
    current_frame: Frame | None = None
    image_path: str | None = None  # Path to input image file
    image: np.ndarray | None = None  # Loaded image data
    camera_id: int | None = None  # Camera device ID for capture
    frames_requested: int | None = None  # Number of frames to capture

    # Edge detection
    contour: Contour | None = None
    detected_contour: Any | None = None
    drop_contour: Any | None = None
    contours_by_frame: list[Contour] = Field(default_factory=list)
    fluid_interface_contour: Contour | None = None
    solid_interface_contour: Contour | None = None

    # Geometry / scaling / physics
    detected_geometry: Geometry | None = None
    geometry: Geometry | None = None  # Current geometric analysis results
    scale: dict[str, float] = Field(default_factory=dict)
    px_per_mm: float | None = None

    # Solver / optimization / outputs
    fit_results: Fit | None = None
    final_output: Result | None = None
    fit: dict[str, Any] | None = None
    physics: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)
    qa: Any = Field(default_factory=dict)

    # Overlay rendering (optional)
    overlay: np.ndarray | None = None  # overlay-only image (BGR)
    preview: np.ndarray | None = None  # base + overlay composited (BGR)

    # Diagnostics
    timings_ms: dict[str, float] = Field(default_factory=dict)
    log: list[str] = Field(default_factory=list)
    error: str | None = None
    status_message: str | None = None

    # Measurement tracking (for results history)
    measurement_id: str | None = None  # Unique identifier (e.g., "20251204_150000_001")
    measurement_sequence: int | None = None  # Sequential number (e.g., 42)

    # Settings for pipeline stages
    preprocessing_settings: PreprocessingSettings | None = None
    edge_detection_settings: EdgeDetectionSettings | None = None
    needle_diameter_mm: float | None = None
    fluid_density_kg_m3: float | None = None
    drop_density_kg_m3: float | None = None

    # Preprocessing state and markers
    preprocessing_markers: MarkerSet | None = None

    # Geometric regions and overlays
    roi: tuple[int, int, int, int] | None = None  # (x, y, width, height)
    detected_roi: tuple[int, int, int, int] | None = None
    needle_rect: tuple[int, int, int, int] | None = None  # (x, y, width, height)
    contact_line: tuple[tuple[int, int], tuple[int, int]] | None = (
        None  # ((x1,y1), (x2,y2))
    )
    substrate_line: tuple[tuple[float, float], tuple[float, float]] | None = (
        None  # ((x1,y1), (x2,y2))
    )
    roi_mask: np.ndarray | None = None  # Binary mask for ROI
    # Fields populated by preprocessing.run (compatibility)
    preprocessed_state: dict[str, Any] | None = None
    preprocessed_settings: dict[str, Any] | None = None
    preprocessed_history: list[Any] | None = None
    preprocessed_roi: np.ndarray | None = None
    preprocessed_mask: np.ndarray | None = None
    preprocessed_scale: tuple[float, float] | None = None
    contact_line_mask: np.ndarray | None = None
    preprocessed: np.ndarray | None = None
    gray: np.ndarray | None = None

    # Specific algorithm variables (found during refactor)
    contact_points: tuple[tuple[int, int], tuple[int, int]] | None = None
    apex_point: tuple[int, int] | None = None
    r_eq_series_px: list | None = None
    centers_px: list | None = None
    r0_eq_px: float | None = None
    pendant_approximation_methods: list[str] | None = None
    pendant_approximator_settings: dict[str, dict[str, Any]] | None = None
    c0_xy: tuple[float, float] | None = None
    h_px: float | None = None
    _sessile_metrics: dict[str, Any] | None = None
    overlay_commands: list | None = None
    smoothing_results: dict[str, Any] | None = None

    # Backwards-compatible single-frame accessors (legacy tests expect `ctx.frame`)
    @property
    def frame(self) -> Any | None:
        # Prefer explicit current_frame, else try first element of frames
        if self.current_frame is not None:
            return self.current_frame
        if self.frames is None:
            return None
        try:
            # frames may be a numpy array or list-like
            return self.frames[0]
        except Exception:
            return self.frames

    @frame.setter
    def frame(self, value: Any | None) -> None:
        # Allow legacy assignment ctx.frame = ndarray; map to frames/current_frame
        if value is None:
            self.frames = None
        elif isinstance(value, Frame):
            self.frames = [value]
        elif isinstance(value, np.ndarray):
            self.frames = [value]
        else:
            self.frames = [cast(np.ndarray, value)]

    def note(self, message: str) -> None:
        self.log.append(message)

    def time(self, stage: str, ms: float) -> None:
        """time.

        Parameters
        ----------
        stage : type
        Description.
        ms : type
        Description.
        """
        self.timings_ms[stage] = ms
