"""Calibration provenance, independent of numerical acceptance thresholds."""

from typing import Literal

from pydantic import BaseModel, Field


class CalibrationProvenance(BaseModel):
    origin: Literal["measured", "manual", "estimated", "missing"]
    px_per_mm: float | None = None
    component_confidence: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def physical_values_enabled(self) -> bool:
        import math

        return (
            self.origin in {"measured", "manual"}
            and self.px_per_mm is not None
            and math.isfinite(self.px_per_mm)
            and self.px_per_mm > 0
        )
