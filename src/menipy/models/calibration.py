"""Calibration provenance, independent of numerical acceptance thresholds."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class CalibrationProvenance(BaseModel):
    origin: Literal["measured", "manual", "estimated", "missing"]
    px_per_mm: float | None = None
    method: Literal[
        "needle_diameter", "known_distance", "direct", "reference", "uncalibrated"
    ] = "uncalibrated"
    reference_id: str | None = None
    reference_geometry: dict[str, Any] = Field(default_factory=dict)
    entered_distance_mm: float | None = None
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
