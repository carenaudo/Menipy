"""Data models for numerical fitting results and configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_numpy.typing import Np1DArrayFp64, NpNDArrayFp64

from .typing import FloatVec


class SolverInfo(BaseModel):
    """Provenance of the numerical solve/fit."""

    backend: Literal["scipy.least_squares", "scipy.leastsq", "custom"] = (
        "scipy.least_squares"
    )
    method: str | None = Field(default=None, description="lm/trf/dogbox/etc.")
    iterations: int | None = Field(default=None, ge=0)
    success: bool = True
    message: str | None = None
    time_ms: float | None = Field(default=None, ge=0)


class Residuals(BaseModel):
    """Basic residual metrics following SciPy least-squares conventions."""

    rmse: float = Field(ge=0)
    max_abs: float = Field(ge=0)
    dof: int | None = Field(default=None, ge=0)
    # Optional full residual vector (e.g., per-point contour error in pixels/mm)
    r: FloatVec | None = None


class Confidence(BaseModel):
    """Optional confidence/uncertainty outputs."""

    covariance: NpNDArrayFp64 | None = None
    param_stderr: Np1DArrayFp64 | None = None
    ci95: list[tuple[float, float]] | None = None  # per-parameter CIs


class Fit(BaseModel):
    """Aggregated results of a numerical fit."""

    params: list[float]
    param_names: list[str] | None = None
    solver: SolverInfo
    residuals: Residuals
    confidence: Confidence | None = None


@dataclass
class FitConfig:
    """Parameters controlling the generic fit wrapper used by common/solver."""

    x0: list[float]  # Initial parameter guess
    bounds: tuple[list[float], list[float]] = field(
        default_factory=lambda: ([-math.inf], [math.inf])
    )
    loss: str = "soft_l1"  # options: linear, huber, cauchy, etc.
    f_scale: float = 1.0
    max_nfev: int | None = None
    verbose: int = 0
    distance: str = "pointwise"  # or "normal_projection"
    weights: list[float] | None = None
    param_names: list[str] | None = None

    def to_solver_args(self) -> dict:
        """to solver args.

        Returns
        -------
        type
        Description.
        """
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
