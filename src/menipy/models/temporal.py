"""Contracts for calibrated image sequences and dynamic sessile analysis."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TemporalState = Literal["advancing", "receding", "pinned", "invalid"]


class SequenceMetadata(BaseModel):
    """Immutable provenance and timing information for an acquired sequence."""

    source_type: Literal["video", "image_sequence", "memory"]
    source_id: str
    sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    timestamps_s: list[float]
    frame_count: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_timing(self) -> SequenceMetadata:
        if len(self.timestamps_s) != self.frame_count:
            raise ValueError("sequence_timestamp_count_mismatch")
        if any(value < 0 for value in self.timestamps_s):
            raise ValueError("sequence_timestamp_negative")
        if any(
            self.timestamps_s[index] <= self.timestamps_s[index - 1]
            for index in range(1, len(self.timestamps_s))
        ):
            raise ValueError("sequence_timestamps_not_monotonic")
        return self


class TemporalFrameResult(BaseModel):
    """Raw scientific result and audit information for one sequence frame."""

    frame_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    accepted: bool = False
    rejection_reasons: list[str] = Field(default_factory=list)
    segment_id: int = 0
    state: TemporalState = "invalid"
    baseline: tuple[tuple[float, float], tuple[float, float]] | None = None
    contacts: tuple[tuple[float, float], tuple[float, float]] | None = None
    theta_left_deg: float | None = None
    theta_right_deg: float | None = None
    half_width_mm: float | None = None
    contact_velocity_mm_s: float | None = None
    contour: list[tuple[float, float]] | None = None
    predicted_roi: tuple[int, int, int, int] | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class DynamicSessileResult(BaseModel):
    """Persistable Phase-D result for a complete dynamic sessile sequence."""

    schema_version: Literal["1.0"] = "1.0"
    pipeline: Literal["sessile_dynamic"] = "sessile_dynamic"
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    metadata: SequenceMetadata
    calibration: dict[str, Any]
    summary: dict[str, Any] = Field(default_factory=dict)
    frames: list[TemporalFrameResult] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
