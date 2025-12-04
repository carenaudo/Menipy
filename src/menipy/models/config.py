"""
Configuration models for pipeline settings and physical parameters.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict

from .unit_types import Density, Length, SurfaceTension


class PhysicsParams(BaseModel):
    """
    Physical parameters required by fits (Young–Laplace, Rayleigh–Lamb, Jurin).
    Values can be provided as strings with units (e.g., "1000 kg/m^3", "72 mN/m").
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    delta_rho: Optional[Density] = Field(default=None, description="Density difference Δρ between the two phases.")
    g: float = Field(default=9.80665, description="Acceleration due to gravity [m/s^2]")
    surface_tension_guess: Optional[SurfaceTension] = Field(default=None)
    needle_radius: Optional[Length] = Field(default=None)
    tube_radius: Optional[Length] = Field(default=None)
    temperature_C: Optional[float] = None


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
    method: Literal["clahe", "histogram", "otsu"] = Field(default="clahe")
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
    convert_to_grayscale: bool = Field(
        default=True, description="Convert the source image/ROI to grayscale before other steps."
    )
    work_on_roi_mask: bool = Field(
        default=True, description="Operate strictly inside ROI mask clone"
    )
    # New: fill holes settings
    class FillHolesSettings(BaseModel):
        """Options to fill interior holes of the ROI and remove small spurious objects near contact line."""

        enabled: bool = Field(default=False, description="Enable filling small holes and removing spurious objects")
        max_hole_area: int = Field(default=500, ge=0, description="Maximum hole area (in pixels) to fill")
        remove_spurious_near_contact: bool = Field(default=True, description="Attempt to remove small objects near the contact line")
        proximity_px: int = Field(default=5, ge=0, description="Pixel proximity to contact line to consider for spurious removal")

    resize: ResizeSettings = Field(default_factory=ResizeSettings)
    filtering: FilterSettings = Field(default_factory=FilterSettings)
    background: BackgroundSettings = Field(default_factory=BackgroundSettings)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    contact_line: ContactLineSettings = Field(default_factory=ContactLineSettings)
    fill_holes: FillHolesSettings = Field(default_factory=FillHolesSettings)
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
