"""Pipeline execution helper for Menipy GUI."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtWidgets import QMainWindow, QMessageBox, QPlainTextEdit

from menipy.gui.controllers.edge_detection_controller import (
    EdgeDetectionPipelineController,
)
from menipy.gui.controllers.preprocessing_controller import (
    PreprocessingPipelineController,
)
from menipy.gui.services.settings_service import pipeline_settings_for
from menipy.pipelines.base import canonical_stage_name

logger = logging.getLogger(__name__)

# Pipeline settings (from the method tabs of the analysis settings dialog or a
# preset) that are passed to a run. A tab input whose key is not listed here
# has no effect on the analysis.
FORWARDED_PIPELINE_SETTINGS = (
    "experimental_geometry_mode",
    "needle_geometry_method",
    "pendant_initializer",
    "pendant_contour_model",
    "contact_angle_method",
    "onnx_proposal_mode",
    "segmentation_provider",
    "onnx_proposal_classes",
    "pendant_approximation_methods",
    "pendant_approximator_settings",
)


class PipelineController:
    """Handles pipeline execution and VM callbacks for the main window."""

    def __init__(
        self,
        window: QMainWindow,
        setup_ctrl,
        preview_panel,
        results_panel,
        preprocessing_ctrl: PreprocessingPipelineController | None,
        edge_detection_ctrl: EdgeDetectionPipelineController | None,
        pipeline_map: Mapping[str, type],
        sops: Any | None,
        run_vm: Any | None,
        log_view: QPlainTextEdit | None,
    ) -> None:
        self.window = window
        self.setup_ctrl = setup_ctrl
        self.preview_panel = preview_panel
        self.results_panel = results_panel
        self.preprocessing_ctrl = preprocessing_ctrl
        self.edge_detection_ctrl = edge_detection_ctrl
        self.sops = sops
        self.run_vm = run_vm
        self.log_view = log_view
        self.pipeline_map = {str(k).lower(): v for k, v in (pipeline_map or {}).items()}
        self.contact_points = None  # Placeholder for manual contact points
        self._latest_acquisition_overlays: dict[str, Any] = {}
        self._completed_jobs = set()
        self._setup_revision = 0
        self._busy_controls = []

    def _collect_acquisition_inputs(self) -> tuple[bool, dict[str, Any]]:
        overlays: dict[str, Any] = {}
        preview = self.preview_panel
        missing: list[str] = []

        roi_rect = preview.roi_rect() if hasattr(preview, "roi_rect") else None
        if not roi_rect:
            missing.append("ROI")
        else:
            overlays["roi"] = roi_rect

        needle_rect = preview.needle_rect() if hasattr(preview, "needle_rect") else None
        if needle_rect:
            overlays["needle_rect"] = needle_rect
        # Ordinary sessile contact-angle analysis needs a contact boundary and
        # visible liquid flanks, not a dispensing needle. Needle-in-drop is a
        # separate analysis mode with its own requirements.
        pipeline_name = str(self.setup_ctrl.gather_run_params().get("name", "")).lower()
        if pipeline_name == "needle_hysteresis" and not needle_rect:
            missing.append("needle region")

        requires_contact = bool(
            getattr(self.window.settings, "acquisition_requires_contact_line", False)
        )
        contact_line = (
            preview.contact_line_segment()
            if hasattr(preview, "contact_line_segment")
            else None
        )
        if requires_contact:
            if not contact_line:
                missing.append("contact line")
            else:
                overlays["contact_line"] = contact_line
        elif contact_line:
            overlays["contact_line"] = contact_line

        if missing:
            QMessageBox.warning(
                self.window,
                "Acquisition Requirements",
                f"Unable to run acquisition. Please define: {', '.join(missing)}.",
            )
            return False, {}

        self._latest_acquisition_overlays = overlays
        return True, overlays

    def _preprocessing_payload(self) -> dict[str, Any]:
        ctrl = self.preprocessing_ctrl
        if not ctrl:
            return {}
        payload: dict[str, Any] = {
            "preprocessing_settings": ctrl.settings.model_copy(deep=True),
        }
        try:
            payload["preprocessing_markers"] = ctrl.markers.model_copy(deep=True)
        except Exception:
            payload["preprocessing_markers"] = ctrl.markers
        return payload

    def _edge_detection_payload(self) -> dict[str, Any]:
        ctrl = self.edge_detection_ctrl
        if not ctrl:
            return {}
        payload: dict[str, Any] = {
            "edge_detection_settings": ctrl.settings.model_copy(deep=True),
        }
        return payload

    def _calibration_result_payload(self) -> dict[str, Any]:
        result = getattr(self.window, "_last_calibration_result", None)
        if result is None:
            return {}

        payload: dict[str, Any] = {}
        if getattr(result, "detector_diagnostics", None):
            payload["detector_diagnostics"] = dict(result.detector_diagnostics)
        if getattr(result, "roi_rect", None):
            payload["roi"] = result.roi_rect
        if getattr(result, "needle_rect", None):
            payload["needle_rect"] = result.needle_rect
        if getattr(result, "drop_contour", None) is not None:
            payload["drop_contour"] = result.drop_contour
        if getattr(result, "contact_points", None):
            payload["contact_points"] = result.contact_points
        if getattr(result, "apex_point", None):
            payload["apex_point"] = result.apex_point
        if getattr(result, "substrate_line", None):
            payload["substrate_line"] = result.substrate_line
        if getattr(result, "substrate_profile", None):
            payload["substrate_profile"] = result.substrate_profile
        if getattr(result, "liquid_geometry", None):
            payload["liquid_geometry"] = result.liquid_geometry
        return payload

    def _build_pipeline_run_kwargs(
        self,
        *,
        sandbox_config: Mapping[str, Any] | None = None,
        auto_calibrate: bool = False,
    ) -> tuple[str, type | None, dict[str, Any], list[str]]:
        params = self.setup_ctrl.gather_run_params()
        name = (params.get("name") or "sessile" or "").lower()
        pipeline_cls = self.pipeline_map.get(name)
        warnings: list[str] = []

        image = params.get("image")
        if name == "sessile_dynamic" and params.get("mode") == "batch":
            image = params.get("batch_folder")
        cam_id = params.get("cam_id")
        frames = params.get("frames")

        overlays: dict[str, Any] = {}
        for key, getter in (
            ("roi", "roi_rect"),
            ("needle_rect", "needle_rect"),
            ("contact_line", "contact_line_segment"),
        ):
            method = getattr(self.preview_panel, getter, None)
            value = method() if callable(method) else None
            if value is not None:
                overlays[key] = value

        calibration_payload = self._calibration_result_payload()
        for key, value in calibration_payload.items():
            overlays.setdefault(key, value)

        sandbox = dict(sandbox_config or {})
        preprocessing_settings = sandbox.get("preprocessing_settings")
        edge_detection_settings = sandbox.get("edge_detection_settings")
        if preprocessing_settings is None:
            overlays.update(self._preprocessing_payload())
        else:
            overlays["preprocessing_settings"] = preprocessing_settings
        if edge_detection_settings is None:
            overlays.update(self._edge_detection_payload())
        else:
            overlays["edge_detection_settings"] = edge_detection_settings

        calibration_params = self.setup_ctrl.get_calibration_params() or {}
        try:
            needle_diameter_mm = float(
                calibration_params.get("needle_diameter_mm", 0.54)
            )
        except (ValueError, TypeError):
            needle_diameter_mm = 0.54
        needle_rect = overlays.get("needle_rect")
        supplied_scale = calibration_params.get("px_per_mm")
        spatial = calibration_params.get("spatial_calibration") or {}
        px_per_mm = None
        scale_method = "uncalibrated"
        if spatial.get("method") == "known_distance":
            try:
                from menipy.common.spatial_calibration import (
                    px_per_mm_from_known_distance,
                )

                px_per_mm = px_per_mm_from_known_distance(
                    spatial.get("segment"), spatial.get("distance_mm")
                )
                scale_method = "known_distance"
            except (TypeError, ValueError):
                warnings.append("Known-distance calibration is invalid; review its segment and distance.")
        elif isinstance(supplied_scale, (int, float)) and np.isfinite(supplied_scale) and supplied_scale > 0:
            px_per_mm = float(supplied_scale)
            scale_method = "direct"
        elif needle_rect and isinstance(needle_diameter_mm, (int, float)) and needle_diameter_mm > 0:
            width = float(needle_rect[2])
            if width > 0:
                px_per_mm = width / needle_diameter_mm
                scale_method = "needle_diameter"
        if px_per_mm is None:
            warnings.append("No spatial calibration is available; physical values will be withheld.")

        run_kwargs = dict(overlays)
        run_kwargs["calibration_params"] = calibration_params
        if px_per_mm is not None:
            run_kwargs["scale"] = {"px_per_mm": px_per_mm}
        calibration_result = getattr(self.window, "_last_calibration_result", None)
        run_kwargs["calibration_provenance"] = {
            "origin": "measured" if scale_method == "needle_diameter" else "manual" if scale_method == "direct" else "missing",
            "method": scale_method,
            "px_per_mm": px_per_mm,
            "reference_id": spatial.get("reference_id"),
            "reference_geometry": {"segment": spatial.get("segment")} if spatial.get("segment") else {},
            "entered_distance_mm": spatial.get("distance_mm"),
            "warnings": list(warnings),
            "component_confidence": dict(
                getattr(calibration_result, "confidence_scores", {}) or {}
            ),
        }
        run_kwargs["physics"] = {
            "rho1": calibration_params.get("drop_density_kg_m3", 1000.0),
            "rho2": calibration_params.get("fluid_density_kg_m3", 1.2),
            "g": calibration_params.get("g", 9.80665),
        }
        settings_owner = getattr(self, "settings", None) or getattr(
            self.window, "settings", None
        )
        pipeline_settings = pipeline_settings_for(settings_owner, name)
        for key in FORWARDED_PIPELINE_SETTINGS:
            if key in pipeline_settings:
                run_kwargs[key] = pipeline_settings[key]
        if params.get("analysis_params"):
            run_kwargs["analysis_params"] = params.get("analysis_params")
        if image is not None:
            run_kwargs["image"] = image
        if cam_id is not None:
            run_kwargs["camera"] = cam_id
        if frames is not None:
            run_kwargs["frames"] = frames

        if auto_calibrate:
            run_kwargs["auto_calibrate"] = True
        if name == "sessile_dynamic":
            run_kwargs.pop("image", None)
            run_kwargs["sequence_path"] = image
            run_kwargs["sequence_fps"] = (params.get("analysis_params") or {}).get(
                "sequence_fps"
            )
            if not needle_rect:
                run_kwargs.pop("scale", None)
                run_kwargs["calibration_provenance"]["origin"] = "missing"
        if image is None and cam_id is None:
            item = getattr(self.preview_panel, "image_item", None)
            if item is not None and hasattr(item, "get_original_image"):
                run_kwargs["image"] = item.get_original_image()
        return name, pipeline_cls, run_kwargs, warnings

    def _should_check_acquisition(self, stages: list[str] | None) -> bool:
        if not stages:
            return True
        return any((stage or "").strip().lower() == "acquisition" for stage in stages)

    def _revision(self):
        import hashlib
        import json

        from menipy.gui.services.pipeline_runner import json_settings

        _, _, parameters, _ = self._build_pipeline_run_kwargs()
        data = json.dumps(json_settings(parameters), sort_keys=True)
        name = self.setup_ctrl.current_pipeline_name()
        return (
            f"{self._setup_revision}:{name}:{hashlib.sha256(data.encode()).hexdigest()}"
        )

    def mark_setup_changed(self, *_):
        self._setup_revision += 1
        if self.run_vm is not None and self.run_vm.busy:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, self._disable_execution_controls)

    def install_execution_controls(self):
        from PySide6.QtWidgets import (
            QAbstractButton,
            QAbstractSpinBox,
            QComboBox,
            QLineEdit,
        )

        stop = getattr(self.window, "actionStop", None)
        if stop is not None:
            stop.setEnabled(False)

        for widget in self.window.setup_panel.findChildren(QLineEdit):
            widget.textChanged.connect(self.mark_setup_changed)
        for widget in self.window.setup_panel.findChildren(QComboBox):
            widget.currentIndexChanged.connect(self.mark_setup_changed)
        for widget in self.window.setup_panel.findChildren(QAbstractSpinBox):
            widget.editingFinished.connect(self.mark_setup_changed)
        for widget in self.window.setup_panel.findChildren(QAbstractButton):
            if widget.isCheckable():
                widget.toggled.connect(self.mark_setup_changed)
        for owner, signals in (
            (self.setup_ctrl, ("pipeline_changed", "source_mode_changed")),
            (self.preprocessing_ctrl, ("settingsChanged", "markersChanged")),
            (self.edge_detection_ctrl, ("settingsChanged",)),
        ):
            for signal in signals:
                if owner is not None and hasattr(owner, signal):
                    getattr(owner, signal).connect(self.mark_setup_changed)
        view = getattr(self.preview_panel, "image_view", None)
        if view is not None:
            view.scene().changed.connect(self.mark_setup_changed)

    def _disable_execution_controls(self):
        from PySide6.QtWidgets import QAbstractButton
        from shiboken6 import isValid

        if not isValid(self.window) or not self.run_vm.busy:
            return
        controls = [
            getattr(self.window, name, None)
            for name in ("actionRunFull", "actionRunSelected")
        ]
        for button in self.window.findChildren(QAbstractButton):
            if any(
                word in button.objectName().lower()
                for word in ("run", "analyze", "autocalibrate", "play")
            ):
                controls.append(button)
        known = {id(control) for control, _ in self._busy_controls}
        for control in controls:
            if control is not None and isValid(control):
                if id(control) not in known:
                    self._busy_controls.append((control, control.isEnabled()))
                control.setEnabled(False)

    def on_state_changed(self, job_id, state):
        from shiboken6 import isValid

        busy = state in ("queued", "running", "stopping")
        if state == "queued":
            self._disable_execution_controls()
        elif not busy:
            for control, enabled in self._busy_controls:
                if isValid(control):
                    control.setEnabled(enabled)
            self._busy_controls = []
        stop = getattr(self.window, "actionStop", None)
        if stop is not None:
            stop.setEnabled(busy and state != "stopping")
        self.window.statusBar().showMessage(state.capitalize() + ("…" if busy else ""))

    def _submit_analysis(
        self,
        *,
        stages=(),
        operation="analysis",
        sandbox_config=None,
        check_acquisition=False,
        auto_calibrate=False,
        use_step_list=False,
    ):
        from menipy.gui.services.pipeline_runner import RunRequest

        try:
            if self.run_vm is None:
                raise RuntimeError("Background execution service is unavailable.")
            if self.run_vm.busy:
                self.window.statusBar().showMessage("An operation is already running.")
                return None
            name, pipeline, parameters, warnings = self._build_pipeline_run_kwargs(
                sandbox_config=sandbox_config, auto_calibrate=auto_calibrate
            )
            if pipeline is None:
                raise ValueError(f"Unknown pipeline: {name}")
            if check_acquisition and name != "sessile_dynamic":
                ready, overlays = self._collect_acquisition_inputs()
                if not ready:
                    return None
                parameters.update(overlays)
            stages = tuple(
                canonical for canonical in map(canonical_stage_name, stages) if canonical
            )
            request = RunRequest.create(
                name,
                parameters,
                operation=operation,
                stages=stages,
                revision=self._revision(),
                warnings=() if auto_calibrate else warnings,
                skip_stages=self._skipped_stages(pipeline) if use_step_list else (),
            )
            return self.run_vm.submit(request)
        except Exception as exc:
            self.on_pipeline_error(f"Could not submit analysis: {exc}")
            return None

    def run_simple_analysis(self):
        return self._submit_analysis(operation="quick_analysis", check_acquisition=True)

    def _skipped_stages(self, pipeline_cls) -> tuple[str, ...]:
        """Optional stages unticked in the step list for ``pipeline_cls``."""
        collect = getattr(self.setup_ctrl, "collect_included_stages", None)
        skipped = getattr(pipeline_cls, "skipped_stages", None)
        if not (self.sops and callable(collect) and callable(skipped)):
            return ()
        return tuple(skipped(collect()))

    def run_full(self):
        return self._submit_analysis(check_acquisition=True, use_step_list=True)

    def run_all(self):
        """Run every stage except the optional ones unticked in the step list."""
        return self._submit_analysis(
            operation="sop", check_acquisition=True, use_step_list=True
        )

    def run_stage(self, stage_name):
        """Run the pipeline up to ``stage_name`` (its prerequisites included)."""
        return self._submit_analysis(
            stages=(stage_name,),
            operation="stage",
            check_acquisition=stage_name == "acquisition",
            use_step_list=True,
        )

    def test_stage(self, stage_name, sandbox_config=None):
        if not stage_name or stage_name == "acquisition":
            return None
        return self._submit_analysis(
            stages=(stage_name,),
            operation="stage_test",
            sandbox_config=sandbox_config,
            auto_calibrate=True,
        )

    def on_completed(self, completion):
        from menipy.models.results import MeasurementResult, build_persisted_analysis

        request = completion.request
        if request.job_id in self._completed_jobs:
            return
        self._completed_jobs.add(request.job_id)
        if request.operation == "calibration":
            return  # Owned by the originating calibration dialog.
        if completion.state == "failed":
            self.on_pipeline_error(completion.error or "Analysis failed.")
            return
        if completion.state != "completed" or completion.ctx is None:
            return
        ctx = completion.ctx
        persisted = build_persisted_analysis(ctx)
        if request.pipeline == "sessile_dynamic":
            persisted["results"].pop("series", None)
        current = request.revision == self._revision()
        measurement = None
        if ctx.results or not persisted["accepted"]:
            metadata = request.metadata()
            metadata["results_schema_version"] = (ctx.results or {}).get(
                "schema_version"
            )
            metadata["runtime"] = ctx.execution_provenance
            metadata["warnings"] = list(completion.warnings)
            # Runtime-derived calibration remains auditable alongside requested settings.
            metadata["effective_calibration"] = {
                "scale": ctx.scale,
                "needle_diameter_mm": ctx.needle_diameter_mm,
                "provenance": ctx.calibration_provenance.model_dump(mode="json")
                if ctx.calibration_provenance
                else None,
            }
            measurement = MeasurementResult(
                id=request.job_id,
                timestamp=request.submitted_at,
                pipeline=request.pipeline,
                file_path=request.source,
                file_name=Path(request.source).name if request.source else None,
                run_metadata=metadata,
                **persisted,
            )
            self.results_panel.add_measurement(measurement, activate=current)
        if current:
            self._display_context(ctx)
        self.append_logs(getattr(ctx, "log", []))
        self.window.statusBar().showMessage(
            (
                "Result kept in memory; history is unsaved."
                if getattr(
                    getattr(self.results_panel, "history", None), "unsaved", False
                )
                else "Analysis complete."
                if current
                else "Result saved to history; setup has changed."
            ),
            4000,
        )

        return measurement

    def _display_context(self, ctx):
        display = getattr(self.preview_panel, "display_context", None)
        if callable(display):
            display(ctx)
        elif getattr(ctx, "preview", None) is not None:
            self.preview_panel.display(ctx.preview)

    def append_logs(self, lines: Any) -> None:
        if not self.log_view:
            return
        try:
            if not lines:
                return
            if isinstance(lines, (list, tuple)):
                for line in lines:
                    self.log_view.appendPlainText(str(line))
            else:
                self.log_view.appendPlainText(str(lines))
        except Exception:
            pass

    def on_pipeline_error(self, message: str) -> None:
        """Handles pipeline errors by showing a dialog and logging the incident."""
        logger.error(f"Pipeline Error: {message}")

        # Use the static method to maintain compatibility with existing test mocks
        display_msg = "An error occurred during the analysis pipeline.\n\n" + message
        QMessageBox.critical(None, "Pipeline Error", display_msg)

        try:
            self.window.statusBar().showMessage(f"Error: {message[:50]}...", 5000)
        except Exception:
            pass
