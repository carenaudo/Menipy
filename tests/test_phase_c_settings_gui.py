"""GUI persistence coverage for opt-in ONNX proposals."""

from __future__ import annotations

from menipy.gui.dialogs.analysis_settings.pendant_settings import (
    PipelineSettingsWidget as PendantSettings,
)
from menipy.gui.dialogs.analysis_settings.sessile_settings import (
    PipelineSettingsWidget as SessileSettings,
)


def test_onnx_proposals_are_off_by_default(qtbot) -> None:
    for widget_type in (SessileSettings, PendantSettings):
        widget = widget_type(settings={})
        qtbot.addWidget(widget)
        settings = widget.get_settings()
        assert settings["onnx_proposal_mode"] == "off"
        assert settings["segmentation_provider"] == "mobilesam"


def test_onnx_shadow_settings_roundtrip(qtbot) -> None:
    for widget_type in (SessileSettings, PendantSettings):
        widget = widget_type(
            settings={
                "onnx_proposal_mode": "shadow",
                "segmentation_provider": "mobilesam",
            }
        )
        qtbot.addWidget(widget)
        settings = widget.get_settings()
        assert settings["onnx_proposal_mode"] == "shadow"
        assert settings["segmentation_provider"] == "mobilesam"

