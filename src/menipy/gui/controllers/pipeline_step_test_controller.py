"""Controller for the scientific pipeline step test panel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QDialog, QMessageBox

from menipy.gui.dialogs.edge_detection_config_dialog import EdgeDetectionConfigDialog
from menipy.gui.dialogs.geometry_config_dialog import GeometryConfigDialog
from menipy.gui.dialogs.overlay_config_dialog import OverlayConfigDialog
from menipy.gui.dialogs.physics_config_dialog import PhysicsConfigDialog
from menipy.gui.dialogs.preprocessing_config_dialog import PreprocessingConfigDialog
from menipy.gui.views.pipeline_step_test_panel import PipelineStepTestPanel
from menipy.models.config import (
    EdgeDetectionSettings,
    PhysicsParams,
    PreprocessingSettings,
)


class PipelineStepTestController(QObject):
    """Owns sandbox settings and test-stage execution for the left rail."""

    EDITABLE_STAGE_HELP = {
        "preprocessing": "Edit preprocessing settings in a sandbox. Apply keeps them.",
        "contour_extraction": "Edit edge detection settings in a sandbox. Apply keeps them.",
        "geometric_features": "Edit geometry settings in a sandbox. Apply keeps them.",
        "physics": "Edit unit-aware physics settings in a sandbox. Apply keeps them.",
        "overlay": "Edit overlay rendering settings in a sandbox. Apply keeps them.",
    }

    def __init__(
        self,
        *,
        window,
        panel: PipelineStepTestPanel,
        setup_ctrl,
        pipeline_ctrl,
        preprocessing_ctrl=None,
        edge_detection_ctrl=None,
        pipeline_map: Mapping[str, type] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.window = window
        self.panel = panel
        self.setup_ctrl = setup_ctrl
        self.pipeline_ctrl = pipeline_ctrl
        self.preprocessing_ctrl = preprocessing_ctrl
        self.edge_detection_ctrl = edge_detection_ctrl
        self.pipeline_map = {str(k).lower(): v for k, v in (pipeline_map or {}).items()}
        self._preprocessing_settings = PreprocessingSettings()
        self._edge_detection_settings = EdgeDetectionSettings()
        self._physics_params = PhysicsParams()
        self._geometry_config: dict[str, Any] = {}
        self._overlay_config: dict[str, Any] = {}
        self._dirty = False

        self._wire_signals()
        self.refresh_from_live()
        self.refresh_stages()

    def refresh_from_live(self) -> None:
        if self.preprocessing_ctrl is not None:
            self._preprocessing_settings = self.preprocessing_ctrl.settings.model_copy(
                deep=True
            )
        else:
            self._preprocessing_settings = PreprocessingSettings()

        if self.edge_detection_ctrl is not None:
            self._edge_detection_settings = (
                self.edge_detection_ctrl.settings.model_copy(deep=True)
            )
        else:
            self._edge_detection_settings = EdgeDetectionSettings()

        physics = getattr(self.window.settings, "physics_config", PhysicsParams())
        self._physics_params = (
            physics.model_copy(deep=True)
            if isinstance(physics, PhysicsParams)
            else PhysicsParams()
        )
        self._geometry_config = dict(
            getattr(self.window.settings, "geometry_config", {}) or {}
        )
        self._overlay_config = dict(
            getattr(self.window.settings, "overlay_config", {}) or {}
        )

        self._set_dirty(False)
        self.panel.set_output("")
        self._update_stage_help()

    def refresh_stages(self) -> None:
        pipeline_name = (self.setup_ctrl.current_pipeline_name() or "sessile").lower()
        pipeline_cls = self.pipeline_map.get(pipeline_name)
        stages: list[str] = []
        if pipeline_cls is not None:
            try:
                pipeline = pipeline_cls()
                stages = [
                    name for name, _fn in pipeline.build_plan() if name != "acquisition"
                ]
            except Exception:
                stages = []
        self.panel.set_stages(stages)
        self.panel.set_status(
            f"Testing {pipeline_name.replace('_', ' ').title()} pipeline."
        )

    def sandbox_config(self) -> dict[str, Any]:
        return {
            "preprocessing_settings": self._preprocessing_settings.model_copy(
                deep=True
            ),
            "edge_detection_settings": self._edge_detection_settings.model_copy(
                deep=True
            ),
            "physics_params": self._physics_params.model_copy(deep=True),
            "geometry_config": dict(self._geometry_config),
            "overlay_config": dict(self._overlay_config),
        }

    def _wire_signals(self) -> None:
        self.panel.runRequested.connect(self.run_stage)
        self.panel.editRequested.connect(self.edit_stage_config)
        self.panel.applyRequested.connect(self.apply_sandbox)
        self.panel.discardRequested.connect(self.discard_sandbox)
        self.panel.stageChanged.connect(lambda _stage: self._update_stage_help())

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = bool(dirty)
        self.panel.set_dirty(self._dirty)

    def _update_stage_help(self) -> None:
        stage = self.panel.current_stage()
        if not stage:
            self.panel.set_stage_help("No stage selected.")
            return
        help_text = self.EDITABLE_STAGE_HELP.get(
            stage,
            "This stage has no dedicated editable test configuration yet. It will run with the current sandbox inputs.",
        )
        self.panel.set_stage_help(help_text)
        self.panel.editConfigBtn.setEnabled(stage in self.EDITABLE_STAGE_HELP)

    @Slot(str)
    def edit_stage_config(self, stage: str) -> None:
        stage = (stage or "").strip().lower()
        if stage == "preprocessing":
            dialog = PreprocessingConfigDialog(
                self._preprocessing_settings, parent=self.window
            )
            if dialog.exec() == QDialog.Accepted:
                self._preprocessing_settings = dialog.settings().model_copy(deep=True)
                self._set_dirty(True)
            return

        if stage == "contour_extraction":
            dialog = EdgeDetectionConfigDialog(
                self._edge_detection_settings, parent=self.window, compact_mode=True
            )
            if dialog.exec() == QDialog.Accepted:
                self._edge_detection_settings = dialog.settings().model_copy(deep=True)
                self._set_dirty(True)
            return

        if stage == "geometric_features":
            dialog = GeometryConfigDialog(parent=self.window)
            dialog.set_config(self._geometry_config)
            if dialog.exec() == QDialog.Accepted:
                self._geometry_config = dict(dialog.get_config())
                self._set_dirty(True)
            return

        if stage == "physics":
            dialog = PhysicsConfigDialog(self._physics_params, parent=self.window)
            if dialog.exec() == QDialog.Accepted:
                self._physics_params = dialog.get_params().model_copy(deep=True)
                self._set_dirty(True)
            return

        if stage == "overlay":
            dialog = OverlayConfigDialog(parent=self.window)
            dialog.set_config(self._overlay_config)
            if dialog.exec() == QDialog.Accepted:
                self._overlay_config = dict(dialog.get_config())
                self._set_dirty(True)
            return

        QMessageBox.information(
            self.window,
            "Step Test",
            f"No editable sandbox configuration is available for '{stage}'.",
        )

    @Slot(str)
    def run_stage(self, stage: str) -> None:
        stage = (stage or "").strip().lower()
        if not stage or stage == "acquisition":
            return
        self.panel.set_status(f"Running {stage.replace('_', ' ').title()}...")
        self.panel.set_output("")
        result = self.pipeline_ctrl.test_stage(stage, self.sandbox_config())
        ctx = result.get("ctx") if isinstance(result, dict) else None
        warnings = result.get("warnings", []) if isinstance(result, dict) else []

        lines: list[str] = []
        if warnings:
            lines.append("Prerequisite warnings:")
            lines.extend(f"- {warning}" for warning in warnings)

        if ctx is not None:
            timings = getattr(ctx, "timings_ms", None)
            if isinstance(timings, dict) and timings:
                lines.append("Timings:")
                lines.extend(
                    f"- {name}: {value:.2f} ms" for name, value in timings.items()
                )
            results = getattr(ctx, "results", None)
            if isinstance(results, dict) and results:
                lines.append("Results:")
                lines.extend(
                    f"- {key}: {value}" for key, value in list(results.items())[:12]
                )
            qa = getattr(ctx, "qa", None)
            if qa:
                lines.append(f"QA: {qa}")
            self.panel.set_status(f"{stage.replace('_', ' ').title()} test complete.")
        else:
            lines.append("Stage test did not return a context.")
            self.panel.set_status(f"{stage.replace('_', ' ').title()} test failed.")

        self.panel.set_output("\n".join(lines).strip())

    @Slot()
    def apply_sandbox(self) -> None:
        if self.preprocessing_ctrl is not None:
            self.preprocessing_ctrl.set_settings(
                self._preprocessing_settings.model_copy(deep=True)
            )
        if self.edge_detection_ctrl is not None:
            self.edge_detection_ctrl.set_settings(
                self._edge_detection_settings.model_copy(deep=True)
            )
        self.window.settings.physics_config = self._physics_params.model_copy(deep=True)
        self.window.settings.geometry_config = dict(self._geometry_config)
        self.window.settings.overlay_config = dict(self._overlay_config)
        preview_panel = getattr(self.window, "preview_panel", None)
        if preview_panel and hasattr(preview_panel, "apply_overlay_config"):
            preview_panel.apply_overlay_config(self._overlay_config)
        try:
            self.window.settings.save()
        except Exception:
            pass
        self._set_dirty(False)
        try:
            self.window.statusBar().showMessage("Test configuration applied.", 1500)
        except Exception:
            pass

    @Slot()
    def discard_sandbox(self) -> None:
        self.refresh_from_live()
        try:
            self.window.statusBar().showMessage("Test configuration discarded.", 1500)
        except Exception:
            pass
