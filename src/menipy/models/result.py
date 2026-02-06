"""Data models for final, high-level analysis outputs."""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_numpy.typing import Np1DArrayFp64

from .fit import Confidence, Residuals, SolverInfo
from .frame import Frame, Calibration
from .geometry import Contour, Geometry
from .config import PhysicsParams
from typing import Literal
from datetime import datetime


class YoungLaplaceFit(BaseModel):
    """Outputs for pendant/sessile ADSA (profile matched to Young–Laplace)."""

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
    """Outputs for oscillating-drop analysis (Rayleigh–Lamb small-amplitude)."""

    gamma_mN_per_m: float = Field(ge=0)
    kinematic_viscosity_mm2_per_s: Optional[float] = Field(default=None, ge=0)
    mode_n: int = Field(default=2, ge=2)
    f0_Hz: float = Field(ge=0, description="natural frequency")
    damping_s_inv: Optional[float] = Field(default=None, ge=0)

    residuals: Residuals
    solver: SolverInfo
    confidence: Optional[Confidence] = None


class CapillaryRiseFit(BaseModel):
    """Outputs for capillary rise (static/dynamic Jurin models)."""

    gamma_mN_per_m: float = Field(ge=0)
    contact_angle_deg: Optional[float] = Field(default=None, ge=0, le=180)
    equilibrium_height_mm: Optional[float] = Field(default=None, ge=0)

    residuals: Residuals
    solver: SolverInfo
    confidence: Optional[Confidence] = None


Result = Union[YoungLaplaceFit, OscillationFit, CapillaryRiseFit]


class AnalysisRecord(BaseModel):
    """
    One complete run of a pipeline stage sequence.
    Stores the minimal artifacts needed for reproducibility.
    """

    kind: Literal["pendant", "sessile", "oscillating", "capillary_rise"]
    frame: Optional[Frame] = None
    contour: Optional[Contour] = None
    geometry: Optional[Geometry] = None
    calibration: Optional[Calibration] = None
    physics: Optional[PhysicsParams] = None

    fit_young_laplace: Optional[YoungLaplaceFit] = None
    fit_oscillation: Optional[OscillationFit] = None
    fit_capillary: Optional[CapillaryRiseFit] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0")

    # simple validator placeholder (keeps previous API)
    @classmethod
    def _validate_kind(cls, k: str) -> str:  # pragma: no cover - trivial
        return k
