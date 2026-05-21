"""Context model for sharing state between pipeline stages."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from .config import EdgeDetectionSettings, PreprocessingSettings
from .frame import Frame
from .geometry import Contour, Geometry
from .fit import Fit
from .result import Result
from .state import MarkerSet


class Context(BaseModel):
    """
    Mutable bag of state shared across pipeline stages.
    Pipelines freely attach fields here. Commonly used keys are predeclared.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Acquisition / images
    frames: Union[np.ndarray, List[np.ndarray], List[Frame], None] = (
        None  # np.ndarray or list[np.ndarray]
    )
    current_frame: Optional[Frame] = None
    image_path: Optional[str] = None  # Path to input image file
    image: Optional[np.ndarray] = None  # Loaded image data
    camera_id: Optional[int] = None  # Camera device ID for capture
    frames_requested: Optional[int] = None  # Number of frames to capture

    # Edge detection
    contour: Optional[Contour] = None
    detected_contour: Optional[Any] = None
    drop_contour: Optional[Any] = None
    contours_by_frame: list[Contour] = Field(default_factory=list)
    fluid_interface_contour: Optional[Contour] = None
    solid_interface_contour: Optional[Contour] = None

    # Geometry / scaling / physics
    detected_geometry: Optional[Geometry] = None
    geometry: Optional[Geometry] = None  # Current geometric analysis results
    scale: dict[str, float] = Field(default_factory=dict)
    px_per_mm: Optional[float] = None

    # Solver / optimization / outputs
    fit_results: Optional[Fit] = None
    final_output: Optional[Result] = None
    fit: Optional[Dict[str, Any]] = None
    physics: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    qa: Any = Field(default_factory=dict)

    # Overlay rendering (optional)
    overlay: Optional[np.ndarray] = None  # overlay-only image (BGR)
    preview: Optional[np.ndarray] = None  # base + overlay composited (BGR)

    # Diagnostics
    timings_ms: dict[str, float] = Field(default_factory=dict)
    log: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    status_message: Optional[str] = None

    # Measurement tracking (for results history)
    measurement_id: Optional[str] = (
        None  # Unique identifier (e.g., "20251204_150000_001")
    )
    measurement_sequence: Optional[int] = None  # Sequential number (e.g., 42)

    # Settings for pipeline stages
    preprocessing_settings: Optional[PreprocessingSettings] = None
    edge_detection_settings: Optional[EdgeDetectionSettings] = None
    needle_diameter_mm: Optional[float] = None
    fluid_density_kg_m3: Optional[float] = None
    drop_density_kg_m3: Optional[float] = None

    # Preprocessing state and markers
    preprocessing_markers: Optional[MarkerSet] = None

    # Geometric regions and overlays
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    detected_roi: Optional[Tuple[int, int, int, int]] = None
    needle_rect: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    contact_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = (
        None  # ((x1,y1), (x2,y2))
    )
    substrate_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = (
        None  # ((x1,y1), (x2,y2))
    )
    roi_mask: Optional[np.ndarray] = None  # Binary mask for ROI
    # Fields populated by preprocessing.run (compatibility)
    preprocessed_state: Optional[Dict[str, Any]] = None
    preprocessed_settings: Optional[Dict[str, Any]] = None
    preprocessed_history: Optional[List[Any]] = None
    preprocessed_roi: Optional[np.ndarray] = None
    preprocessed_mask: Optional[np.ndarray] = None
    preprocessed_scale: Optional[Tuple[float, float]] = None
    contact_line_mask: Optional[np.ndarray] = None
    preprocessed: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None

    # Specific algorithm variables (found during refactor)
    contact_points: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    apex_point: Optional[Tuple[int, int]] = None
    r_eq_series_px: Optional[list] = None
    centers_px: Optional[list] = None
    r0_eq_px: Optional[float] = None
    pendant_approximation_methods: Optional[list[str]] = None
    pendant_approximator_settings: Optional[Dict[str, Dict[str, Any]]] = None
    c0_xy: Optional[Tuple[float, float]] = None
    h_px: Optional[float] = None
    _sessile_metrics: Optional[Dict[str, Any]] = None
    overlay_commands: Optional[list] = None
    smoothing_results: Optional[Dict[str, Any]] = None

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
        try:
            # If value is a single image, store as frames list
            self.frames = [value]
        except Exception:
            self.frames = value  # type: ignore[assignment]

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
