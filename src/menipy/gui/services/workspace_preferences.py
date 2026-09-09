"""Validated portable display/history preferences; no analysis configuration."""

from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class WorkspacePreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    unit_system: Literal["SI", "CGS"] = "SI"
    show_mode_labels: StrictBool = False
    compare_methods_visible: StrictBool = False
    diagnostics_visible: StrictBool = False
    history_limit: int = Field(default=100, ge=10, le=1000, strict=True)
    results_hidden_columns: dict[str, list[str]] = Field(default_factory=dict)

    @classmethod
    def capture(cls, settings):
        return cls(
            **{
                key: getattr(settings, key)
                for key in cls.model_fields
                if key != "schema_version"
            }
        )

    def apply(self, settings):
        """Persist before changing the shared settings instance."""
        values = self.model_dump(exclude={"schema_version"})
        replace(settings, **values).save()
        for key, value in values.items():
            setattr(settings, key, value)
