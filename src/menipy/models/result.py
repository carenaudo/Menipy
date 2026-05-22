"""Data models for final, high-level analysis outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Union

from pydantic import BaseModel, Field
from pydantic_numpy.typing import Np1DArrayFp64

from .config import PhysicsParams
from .fit import Confidence, Residuals, SolverInfo
from .frame import Calibration, Frame
from .geometry import Contour, Geometry


class YoungLaplaceFit(BaseModel):
    """Outputs for pendant/sessile ADSA (profile matched to Young–Laplace)."""

    gamma_mN_per_m: float = Field(ge=0, description="surface/interfacial tension")
    contact_angle_deg: float | None = Field(default=None, ge=0, le=180)
    apex_radius_mm: float | None = Field(default=None, ge=0)
    drop_volume_uL: float | None = Field(default=None, ge=0)
    # Fitted parameter vector and names (e.g., [γ, R0, ...])
    params: Np1DArrayFp64 | None = None
    param_names: list[str] | None = None

    residuals: Residuals
    solver: SolverInfo
    confidence: Confidence | None = None


class OscillationFit(BaseModel):
    """Outputs for oscillating-drop analysis (Rayleigh–Lamb small-amplitude)."""

    gamma_mN_per_m: float = Field(ge=0)
    kinematic_viscosity_mm2_per_s: float | None = Field(default=None, ge=0)
    mode_n: int = Field(default=2, ge=2)
    f0_Hz: float = Field(ge=0, description="natural frequency")
    damping_s_inv: float | None = Field(default=None, ge=0)

    residuals: Residuals
    solver: SolverInfo
    confidence: Confidence | None = None


class CapillaryRiseFit(BaseModel):
    """Outputs for capillary rise (static/dynamic Jurin models)."""

    gamma_mN_per_m: float = Field(ge=0)
    contact_angle_deg: float | None = Field(default=None, ge=0, le=180)
    equilibrium_height_mm: float | None = Field(default=None, ge=0)

    residuals: Residuals
    solver: SolverInfo
    confidence: Confidence | None = None


Result = Union[YoungLaplaceFit, OscillationFit, CapillaryRiseFit]


class AnalysisRecord(BaseModel):
    """
    One complete run of a pipeline stage sequence.
    Stores the minimal artifacts needed for reproducibility.
    """

    kind: Literal["pendant", "sessile", "oscillating", "capillary_rise"]
    frame: Frame | None = None
    contour: Contour | None = None
    geometry: Geometry | None = None
    calibration: Calibration | None = None
    physics: PhysicsParams | None = None

    fit_young_laplace: YoungLaplaceFit | None = None
    fit_oscillation: OscillationFit | None = None
    fit_capillary: CapillaryRiseFit | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0")

    # simple validator placeholder (keeps previous API)
    @classmethod
    def _validate_kind(cls, k: str) -> str:  # pragma: no cover - trivial
        return k
