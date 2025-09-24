# src/adsa/models/datatypes.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import numpy.typing as npt
from pydantic_numpy.typing import (
    Np2DArrayUint8,
    Np3DArrayUint8,
    Np2DArrayFp64,
    Np1DArrayFp64,
    NpNDArrayFp64,
)
import math
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dataclasses import dataclass, field
# ---- Type aliases -----------------------------------------------------------

# Image arrays: grayscale (H, W) or color (H, W, 3), uint8 typical for OpenCV.
ImageGray = Np2DArrayUint8
ImageBGR = Np3DArrayUint8
ImageAny = Union[ImageGray, ImageBGR]  # validated by shape checks
# Contours: N x 2 (x, y) in pixels or metric units depending on context
ContourArray = Np2DArrayFp64
# Time series: 1-D float arrays (seconds, value)
FloatVec = Np1DArrayFp64

# ---- Camera / acquisition metadata -----------------------------------------

class CameraMeta(BaseModel):
    """Minimal camera metadata recorded with frames."""

    device: Optional[str] = Field(default=None, description="Camera name/id")
    fps: Optional[float] = Field(default=None, gt=0)
    exposure_ms: Optional[float] = Field(default=None, gt=0)
    resolution: Optional[Tuple[int, int]] = Field(
        default=None, description="(height, width)"
    )
    lens_mm: Optional[float] = Field(default=None, gt=0)
    note: Optional[str] = None

class Calibration(BaseModel):
    """Pixel-to-metric scaling and optional distortion parameters."""

    pixels_per_mm: Optional[float] = Field(default=None, gt=0)
    mm_per_pixel: Optional[float] = Field(default=None, gt=0)
    # If both provided, they are cross-checked for consistency
    k1: Optional[float] = None  # radial distortion (optional)
    k2: Optional[float] = None
    cx_cy: Optional[Tuple[float, float]] = Field(
        default=None, description="principal point (cx, cy) in pixels"
    )

    @field_validator("mm_per_pixel")
    @classmethod
    def _invertible_scale(cls, v, info):
        ppx = info.data.get("pixels_per_mm")
        if v and ppx and not np.isclose(v, 1.0 / ppx):
            raise ValueError("mm_per_pixel must equal 1 / pixels_per_mm")
        return v

# ---- Core data records ------------------------------------------------------

class Frame(BaseModel):
    """
    A single image frame (silhouette or raw) plus timing and metadata.
    Accepts grayscale (H, W) or BGR (H, W, 3) uint8 arrays as used by OpenCV. 
    """

    image: "ImageAny" = Field(description="np.uint8 image; shape (H,W) or (H,W,3)")
    timestamp: Optional[datetime] = Field(default=None)
    ms_from_start: Optional[float] = Field(default=None, ge=0)
    camera: Optional[CameraMeta] = None
    calibration: Optional[Calibration] = None

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
    """
    Geometric landmarks required by solvers.
    """

    apex_xy: Optional[Tuple[float, float]] = None     # pendant/sessile
    axis_x: Optional[float] = None                    # symmetry axis x (px or mm)
    baseline_y: Optional[float] = None                # sessile: substrate y
    contact_region_px: Optional[Tuple[int, int]] = None  # index range around CL
    tilt_deg: float = Field(default=0.0)


@dataclass
class Context:
    """
    Mutable bag of state shared across pipeline stages.
    Pipelines freely attach fields here. Commonly used keys are predeclared.
    """
    # Acquisition / images
    roi: Optional[Tuple[int, int, int, int]] = None
    needle_rect: Optional[Tuple[int, int, int, int]] = None
    contact_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    frames: Any | None = None                 # np.ndarray or list[np.ndarray]
    # Edge detection
    contour: Any | None = None                # object with .xy (Nx2) or similar
    contours_by_frame: Any | None = None      # list of contour-like objects
    fluid_interface_contour: Any | None = None
    solid_interface_contour: Any | None = None
    # Geometry / scaling / physics
    geometry: Dict[str, Any] | None = None
    scale: Dict[str, float] | None = None
    physics: Dict[str, Any] | None = None
    # Solver / optimization / outputs
    fit: Dict[str, Any] | None = None
    results: Dict[str, Any] | None = None
    qa: Dict[str, Any] | None = None
    # Overlay rendering (optional)
    overlay: Any | None = None                # overlay-only image (BGR)
    preview: Any | None = None                # base + overlay composited (BGR)
    # Diagnostics
    timings_ms: Dict[str, float] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    error: Optional[str] = None

    # Settings for pipeline stages
    preprocessing_settings: Optional[PreprocessingSettings] = None
    edge_detection_settings: Optional[EdgeDetectionSettings] = None

    def note(self, message: str) -> None:
        self.log.append(message)

    def time(self, stage: str, ms: float) -> None:
        self.timings_ms[stage] = ms

class PhysicsParams(BaseModel):
    """
    Physical parameters required by fits (Young–Laplace, Rayleigh–Lamb, Jurin).
    """

    delta_rho: Optional[float] = Field(default=None, description="density difference Δρ [kg/m^3]")
    g: float = Field(default=9.80665, description="gravity [m/s^2]")
    surface_tension_guess: Optional[float] = Field(default=None, ge=0)
    needle_radius_mm: Optional[float] = Field(default=None, ge=0)
    tube_radius_mm: Optional[float] = Field(default=None, ge=0)
    temperature_C: Optional[float] = None

# ---- Fit parameter/metric containers ---------------------------------------

class SolverInfo(BaseModel):
    """Provenance of the numerical solve/fit."""
    backend: Literal["scipy.least_squares", "scipy.leastsq", "custom"] = "scipy.least_squares"
    method: Optional[str] = Field(default=None, description="lm/trf/dogbox/etc.")
    iterations: Optional[int] = Field(default=None, ge=0)
    success: bool = True
    message: Optional[str] = None
    time_ms: Optional[float] = Field(default=None, ge=0)

class Residuals(BaseModel):
    """Basic residual metrics following SciPy least-squares conventions."""
    rmse: float = Field(ge=0)
    max_abs: float = Field(ge=0)
    dof: Optional[int] = Field(default=None, ge=0)
    # Optional full residual vector (e.g., per-point contour error in pixels/mm)
    r: Optional["FloatVec"] = None

class Confidence(BaseModel):
    """Optional confidence/uncertainty outputs."""

    covariance: Optional[NpNDArrayFp64] = None
    param_stderr: Optional[Np1DArrayFp64] = None
    ci95: Optional[List[Tuple[float, float]]] = None  # per-parameter CIs


@dataclass
class FitConfig:
    """Parameters controlling the generic fit wrapper used by common/solver."""
    x0: list[float]                             # Initial parameter guess
    bounds: tuple[list[float], list[float]]     = field(default_factory=lambda: ([-math.inf], [math.inf]))
    loss: str                                   = "soft_l1"  # options: linear, huber, cauchy, etc.
    f_scale: float                              = 1.0
    max_nfev: Optional[int]                     = None
    verbose: int                                = 0
    distance: str                               = "pointwise"  # or "normal_projection"
    weights: Optional[list[float]]              = None
    param_names: Optional[list[str]]            = None

    def to_solver_args(self) -> dict:
        # Helper to convert to args expected by common/solver.run
        return {
            "x0": self.x0,
            "bounds": self.bounds,
            "loss": self.loss,
            "f_scale": self.f_scale,
            "max_nfev": self.max_nfev,
            "verbose": self.verbose,
            "distance": self.distance,
            "weights": self.weights,
            "param_names": self.param_names,
        }

# --- Specific fit results ----------------------------------------------------

class YoungLaplaceFit(BaseModel):
    """
    Outputs for pendant/sessile ADSA (profile matched to Young–Laplace).
    """

    gamma_mN_per_m: float = Field(ge=0, description="surface/interfacial tension")
    contact_angle_deg: Optional[float] = Field(default=None, ge=0, le=180)
    apex_radius_mm: Optional[float] = Field(default=None, ge=0)
    drop_volume_uL: Optional[float] = Field(default=None, ge=0)
    # Fitted parameter vector and names (e.g., [γ, R0, ...])
    params: Optional[Np1DArrayFp64] = None
    param_names: Optional[List[str]] = None

    residuals: Residuals
    solver: SolverInfo
    confidence: Optional[Confidence] = None

class OscillationFit(BaseModel):
    """
    Outputs for oscillating-drop analysis (Rayleigh–Lamb small-amplitude).
    """

    gamma_mN_per_m: float = Field(ge=0)
    kinematic_viscosity_mm2_per_s: Optional[float] = Field(default=None, ge=0)
    mode_n: int = Field(default=2, ge=2)
    f0_Hz: float = Field(ge=0, description="natural frequency")
    damping_s_inv: Optional[float] = Field(default=None, ge=0)

    residuals: Residuals
    solver: SolverInfo
    confidence: Optional[Confidence] = None

class CapillaryRiseFit(BaseModel):
    """
    Outputs for capillary rise (static/dynamic Jurin models).
    """

    gamma_mN_per_m: float = Field(ge=0)
    contact_angle_deg: Optional[float] = Field(default=None, ge=0, le=180)
    equilibrium_height_mm: Optional[float] = Field(default=None, ge=0)

    residuals: Residuals
    solver: SolverInfo
    confidence: Optional[Confidence] = None


# ---- Preprocessing configuration -----------------------------------------

class ResizeSettings(BaseModel):
    """Parameters controlling optional ROI resizing."""

    enabled: bool = Field(default=False)
    target_width: Optional[int] = Field(default=None, ge=1)
    target_height: Optional[int] = Field(default=None, ge=1)
    preserve_aspect: bool = Field(default=True)
    interpolation: Literal["nearest", "linear", "cubic", "area", "lanczos"] = Field(
        default="linear", description="OpenCV interpolation mode"
    )

    @property
    def has_target(self) -> bool:
        return (self.target_width or self.target_height) is not None


class FilterSettings(BaseModel):
    """Denoising / smoothing options applied within the ROI."""

    enabled: bool = Field(default=False)
    method: Literal["none", "gaussian", "median", "bilateral"] = Field(
        default="none"
    )
    kernel_size: int = Field(default=3, ge=1, description="Odd window size")
    sigma: float = Field(default=1.0, ge=0.0, description="Gaussian sigma")
    sigma_color: float = Field(default=75.0, ge=0.0, description="Bilateral color sigma")
    sigma_space: float = Field(default=75.0, ge=0.0, description="Bilateral spatial sigma")


class BackgroundSettings(BaseModel):
    """Background subtraction configuration confined to ROI mask."""

    enabled: bool = Field(default=False)
    mode: Literal["flat", "rolling_ball"] = Field(default="flat")
    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    rolling_radius: int = Field(default=15, ge=1)


class NormalizationSettings(BaseModel):
    """Local contrast normalization configuration."""

    enabled: bool = Field(default=False)
    method: Literal["clahe", "histogram"] = Field(default="clahe")
    clip_limit: float = Field(default=2.0, ge=0.0)
    grid_size: int = Field(default=8, ge=1)


class ContactLineSettings(BaseModel):
    """Defines how the pipeline treats contact line pixels."""

    strategy: Literal["preserve", "attenuate", "mask"] = Field(default="preserve")
    threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    dilation: int = Field(default=3, ge=1)


class PreprocessingSettings(BaseModel):
    """Aggregated user-configurable preprocessing settings."""

    crop_to_roi: bool = Field(default=True)
    work_on_roi_mask: bool = Field(
        default=True, description="Operate strictly inside ROI mask clone"
    )
    resize: ResizeSettings = Field(default_factory=ResizeSettings)
    filtering: FilterSettings = Field(default_factory=FilterSettings)
    background: BackgroundSettings = Field(default_factory=BackgroundSettings)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    contact_line: ContactLineSettings = Field(default_factory=ContactLineSettings)
    preset_id: Optional[str] = Field(default=None)


class EdgeDetectionSettings(BaseModel):
    """Parameters controlling the edge detection stage."""

    enabled: bool = Field(default=True)
    method: Literal["canny", "sobel", "scharr", "laplacian", "threshold", "active_contour"] = Field(
        default="canny", description="Edge detection algorithm to use"
    )
    # Common preprocessing for edge detection
    gaussian_blur_before: bool = Field(default=True, description="Apply Gaussian blur before edge detection")
    gaussian_kernel_size: int = Field(default=5, ge=1, multiple_of=2, description="Kernel size for Gaussian blur (must be odd)")
    gaussian_sigma_x: float = Field(default=0.0, ge=0.0, description="Gaussian kernel standard deviation in X direction")

    # Canny specific parameters
    canny_threshold1: int = Field(default=50, ge=0, le=255, description="First threshold for the hysteresis procedure")
    canny_threshold2: int = Field(default=150, ge=0, le=255, description="Second threshold for the hysteresis procedure")
    canny_aperture_size: Literal[3, 5, 7] = Field(default=3, description="Aperture size for the Sobel operator")
    canny_L2_gradient: bool = Field(default=False, description="Flag indicating whether a more accurate L2 gradient magnitude should be used")

    # Threshold specific parameters
    threshold_value: int = Field(default=128, ge=0, le=255, description="Threshold value")
    threshold_max_value: int = Field(default=255, ge=0, le=255, description="Maximum value to use with THRESH_BINARY and THRESH_BINARY_INV")
    threshold_type: Literal["binary", "binary_inv", "trunc", "to_zero", "to_zero_inv"] = Field(
        default="binary", description="Type of thresholding to apply"
    )

    # Sobel/Scharr/Laplacian specific parameters (can share some with Canny)
    sobel_kernel_size: int = Field(default=3, ge=1, multiple_of=2, description="Kernel size for Sobel/Scharr (must be odd)")
    laplacian_kernel_size: int = Field(default=1, ge=1, multiple_of=2, description="Kernel size for Laplacian (must be odd)")

    # Active Contour specific parameters (if implemented)
    active_contour_iterations: int = Field(default=100, ge=1, description="Number of iterations for active contour model")
    active_contour_alpha: float = Field(default=0.01, ge=0.0, description="Weight of the contour length term")
    active_contour_beta: float = Field(default=0.1, ge=0.0, description="Weight of the contour smoothness term")

    # Contour refinement and filtering
    min_contour_length: int = Field(default=10, ge=0, description="Minimum length of a detected contour to be considered valid")
    max_contour_length: int = Field(default=10000, ge=0, description="Maximum length of a detected contour to be considered valid")

    # Interface specific settings
    detect_fluid_interface: bool = Field(default=True, description="Detect the droplet-fluid interface")
    detect_solid_interface: bool = Field(default=True, description="Detect the droplet-solid interface")
    solid_interface_proximity: int = Field(default=10, ge=0, description="Pixels from contact line to search for solid interface")

    @field_validator("gaussian_kernel_size", "sobel_kernel_size", "laplacian_kernel_size")
    @classmethod
    def _must_be_odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")
        return v


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


# ---- Aggregated analysis record --------------------------------------------

class AnalysisRecord(BaseModel):
    """ 
    One complete run of a pipeline stage sequence.
    Stores the minimal artifacts needed for reproducibility.
    """

    kind: Literal["pendant", "sessile", "oscillating", "capillary_rise"]
    frame: Optional[Frame] = None             # single image or representative frame
    contour: Optional[Contour] = None
    geometry: Optional[Geometry] = None
    calibration: Optional[Calibration] = None
    physics: Optional[PhysicsParams] = None

    fit_young_laplace: Optional[YoungLaplaceFit] = None
    fit_oscillation: Optional[OscillationFit] = None
    fit_capillary: Optional[CapillaryRiseFit] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0")

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, k: str) -> str:
        return k

# ---- Convenience constructors ----------------------------------------------

def make_frame(image: np.ndarray,
               timestamp: Optional[datetime] = None,
               ms_from_start: Optional[float] = None,
               camera: Optional[CameraMeta] = None,
               calibration: Optional[Calibration] = None) -> Frame:
    """Helper to create a validated Frame."""
    return Frame(image=image,
                 timestamp=timestamp,
                 ms_from_start=ms_from_start,
                 camera=camera,
                 calibration=calibration)

def make_contour(xy: np.ndarray,
                 closed: bool = True,
                 units: Literal["px", "mm"] = "px",
                 smoothing: Optional[float] = None,
                 origin_hint: Optional[Tuple[float, float]] = None) -> Contour:
    """Helper to create a validated Contour."""
    return Contour(xy=xy, closed=closed, units=units,
                   smoothing=smoothing, origin_hint=origin_hint)