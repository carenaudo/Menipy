"""
Context model for sharing state between pipeline stages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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
    model_config = ConfigDict(extra='allow')

    # Acquisition / images
    frames: Any | None = None  # np.ndarray or list[np.ndarray]
    current_frame: Optional[Frame] = None
    image_path: Optional[str] = None  # Path to input image file
    image: Any | None = None  # Loaded image data
    camera_id: Optional[int] = None  # Camera device ID for capture
    frames_requested: Optional[int] = None  # Number of frames to capture

    # Edge detection
    contour: Optional[Contour] = None
    contours_by_frame: list[Contour] = Field(default_factory=list)
    fluid_interface_contour: Optional[Contour] = None
    solid_interface_contour: Optional[Contour] = None

    # Geometry / scaling / physics
    detected_geometry: Optional[Geometry] = None
    geometry: Optional[Geometry] = None  # Current geometric analysis results
    scale: dict[str, float] = Field(default_factory=dict)

    # Solver / optimization / outputs
    fit_results: Optional[Fit] = None
    final_output: Optional[Result] = None
    fit: Optional[Dict[str, Any]] = None
    physics: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    qa: dict[str, Any] = Field(default_factory=dict)

    # Overlay rendering (optional)
    overlay: Any | None = None  # overlay-only image (BGR)
    preview: Any | None = None  # base + overlay composited (BGR)

    # Diagnostics
    timings_ms: dict[str, float] = Field(default_factory=dict)
    log: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    # Measurement tracking (for results history)
    measurement_id: Optional[str] = (
        None  # Unique identifier (e.g., "20251204_150000_001")
    )
    measurement_sequence: Optional[int] = None  # Sequential number (e.g., 42)

    # Settings for pipeline stages
    preprocessing_settings: Optional[PreprocessingSettings] = None
    edge_detection_settings: Optional[EdgeDetectionSettings] = None

    # Preprocessing state and markers
    preprocessing_markers: Optional[MarkerSet] = None

    # Geometric regions and overlays
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    needle_rect: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    contact_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = (
        None  # ((x1,y1), (x2,y2))
    )
    substrate_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = (
        None  # ((x1,y1), (x2,y2))
    )
    roi_mask: Any | None = None  # Binary mask for ROI
    # Fields populated by preprocessing.run (compatibility)
    preprocessed_state: Any | None = None
    preprocessed_settings: Any | None = None
    preprocessed_history: Any | None = None
    preprocessed_roi: Any | None = None
    preprocessed_mask: Any | None = None
    preprocessed_scale: Any | None = None
    contact_line_mask: Any | None = None
    preprocessed: Any | None = None

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
            self.frames = value

    def note(self, message: str) -> None:
        self.log.append(message)

    def time(self, stage: str, ms: float) -> None:
        self.timings_ms[stage] = ms
