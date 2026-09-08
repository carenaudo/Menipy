"""Versioned, typed analysis setup; input images and source paths are excluded."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import EdgeDetectionSettings, PreprocessingSettings
from .state import MarkerSet


class AnalysisPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    pipeline: str
    application_version: str
    stages: list[str]
    unit_system: Literal["SI", "CGS"] = "SI"
    preprocessing: PreprocessingSettings
    edge_detection: EdgeDetectionSettings
    markers: MarkerSet = Field(default_factory=MarkerSet)
    controls: dict[str, float | str]
    pipeline_settings: dict[str, Any] = Field(default_factory=dict)
    geometry: dict[str, Any] = Field(default_factory=dict)
    plugins: dict[str, str] = Field(default_factory=dict)
