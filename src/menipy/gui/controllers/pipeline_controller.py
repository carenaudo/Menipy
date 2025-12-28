"""Pipeline execution helper for Menipy GUI."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Mapping, Optional, Dict

from menipy.gui.controllers.preprocessing_controller import (
    PreprocessingPipelineController,
)
from menipy.gui.controllers.edge_detection_controller import (
    EdgeDetectionPipelineController,
)
from PySide6.QtWidgets import QMessageBox, QPlainTextEdit, QMainWindow

from menipy.models.config import PhysicsParams


class PipelineController:
    """Handles pipeline execution and VM callbacks for the main window."""

    def __init__(
        self,
        window: QMainWindow,
        setup_ctrl,
        preview_panel,
        results_panel,
        preprocessing_ctrl: Optional[PreprocessingPipelineController],
        edge_detection_ctrl: Optional[EdgeDetectionPipelineController],
        pipeline_map: Mapping[str, type],
        sops: Optional[Any],
        run_vm: Optional[Any],
        log_view: Optional[QPlainTextEdit],
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
        self._latest_acquisition_overlays: Dict[str, Any] = {}

    def _collect_acquisition_inputs(self) -> tuple[bool, Dict[str, Any]]:
        overlays: Dict[str, Any] = {}
        preview = self.preview_panel
        missing: list[str] = []

        roi_rect = preview.roi_rect() if hasattr(preview, "roi_rect") else None
        if not roi_rect:
            missing.append("ROI")
        else:
            overlays["roi"] = roi_rect

        needle_rect = preview.needle_rect() if hasattr(preview, "needle_rect") else None
        if not needle_rect:
            missing.append("needle region")
        else:
            overlays["needle_rect"] = needle_rect

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

    def _preprocessing_payload(self) -> Dict[str, Any]:
        ctrl = self.preprocessing_ctrl
        if not ctrl:
            return {}
        payload: Dict[str, Any] = {
            "preprocessing_settings": ctrl.settings.model_copy(deep=True),
        }
        try:
            payload["preprocessing_markers"] = ctrl.markers.model_copy(deep=True)
        except Exception:
            payload["preprocessing_markers"] = ctrl.markers
        return payload

    def _edge_detection_payload(self) -> Dict[str, Any]:
        ctrl = self.edge_detection_ctrl
        if not ctrl:
            return {}
        payload: Dict[str, Any] = {
            "edge_detection_settings": ctrl.settings.model_copy(deep=True),
        }
        return payload

    def _should_check_acquisition(self, stages: Optional[list[str]]) -> bool:
        if not stages:
            return True
        return any((stage or "").strip().lower() == "acquisition" for stage in stages)

    def _get_analysis_image(self):
        """Validates and returns the cropped image and its ROI."""
        if (
            self.preview_panel.image_item is None
            or self.preview_panel.image_item.pixmap().isNull()
        ):
            self.window.statusBar().showMessage("No image loaded.", 3000)
            return None, None

        drop_roi_item = self.preview_panel.roi_item()
        drop_roi_rect = drop_roi_item.rect() if drop_roi_item else None
        if not drop_roi_rect:
            self.window.statusBar().showMessage("Please define a drop ROI first.", 3000)
            return None, None

        image = self.preview_panel.image_item.get_original_image()
        cropped_image = image[
            int(drop_roi_rect.top()) : int(drop_roi_rect.bottom()),
            int(drop_roi_rect.left()) : int(drop_roi_rect.right()),
        ]
        return cropped_image, drop_roi_rect

    def _load_pipeline_modules(self, mode: str) -> dict:
        """Dynamically loads and returns the modules for the given pipeline mode."""
        geometry_module = importlib.import_module(
            ".geometry", package=f"menipy.pipelines.{mode}"
        )
        drawing_module = importlib.import_module(
            ".drawing", package=f"menipy.pipelines.{mode}"
        )

        # The actual 'analyze' function needs to be implemented in the pipeline files.
        # For now, we assume it exists.
        analyze_func = getattr(geometry_module, "analyze", None)
        if analyze_func is None:
            raise AttributeError(
                f"'analyze' function not found in {geometry_module.__name__}"
            )

        return {
            "analyze_func": analyze_func,
            "draw_func": getattr(drawing_module, f"draw_{mode}_overlay"),
            "helper_bundle_class": getattr(geometry_module, "HelperBundle"),
        }

    def _prepare_helper_bundle(self, mode: str, helper_bundle_class, roi_rect):
        """Prepares and returns the mode-specific HelperBundle."""
        # Get the unit-aware physics parameters from settings
        physics_params: PhysicsParams = getattr(
            self.window.settings, "physics_config", PhysicsParams()
        )

        # Get scale from the calibration controller/dialog (this part needs to be located and updated)
        # For now, we'll assume a placeholder value.
        px_per_mm = 100.0  # Placeholder - this needs to be sourced from the calibration UI/settings

        # Convert unit-aware params to SI floats for the backend
        try:
            delta_rho_si = (
                physics_params.delta_rho.to("kg/m**3").m
                if physics_params.delta_rho
                else 1000.0
            )
            needle_diam_mm_si = (
                physics_params.needle_radius.to("mm").m * 2
                if physics_params.needle_radius
                else None
            )
        except Exception as e:
            QMessageBox.critical(
                self.window,
                "Physics Parameter Error",
                f"Could not convert physics parameters to SI units: {e}",
            )
            return None

        if mode == "pendant":
            if needle_diam_mm_si is None:
                QMessageBox.warning(
                    self.window,
                    "Missing Parameter",
                    "Needle radius must be defined for pendant drop analysis.",
                )
                return None
            return helper_bundle_class(
                px_per_mm=px_per_mm,
                needle_diam_mm=needle_diam_mm_si,
                delta_rho=delta_rho_si,
            )

        if mode == "sessile":
            substrate_line_item = self.preview_panel.substrate_line_item()
            substrate_line = (
                substrate_line_item.get_line_in_scene() if substrate_line_item else None
            )
            if substrate_line:
                p1 = substrate_line.p1() - roi_rect.topLeft()
                p2 = substrate_line.p2() - roi_rect.topLeft()
                substrate_line = ((p1.x(), p1.y()), (p2.x(), p2.y()))

            contact_points = self.contact_points if self.contact_points else None
            return helper_bundle_class(
                px_per_mm=px_per_mm,
                substrate_line=substrate_line,
                contact_points=contact_points,
                delta_rho=delta_rho_si,
            )

        QMessageBox.warning(
            self.window, "Unsupported Mode", f"Analysis mode '{mode}' is not supported."
        )
        return None

    def _run_analysis_and_update_ui(
        self, mode, analyze_func, draw_func, image, helpers
    ):
        """Runs the analysis, draws overlays, and updates the GUI."""
        try:
            # 1. Run the analysis
            metrics = analyze_func(image, helpers)

            # 2. Draw overlays on the image
            # The original pixmap is on the main image_item
            original_pixmap = self.preview_panel.image_item.pixmap()
            pixmap_with_overlays = draw_func(original_pixmap, metrics)
            self.preview_panel.display(pixmap_with_overlays)

            # 3. Update the results panel

            # 4. Add to measurement history
            from menipy.models.context import Context

            ctx = Context()
            ctx.results = metrics.derived
            self.add_measurement_to_history(ctx, mode)

            self.window.statusBar().showMessage("Analysis complete.", 3000)
        except Exception as e:
            self.on_pipeline_error(f"An error occurred during analysis: {e}")

    # ------------------------------------------------------------------
    # Slots wired by MainWindow
    # ------------------------------------------------------------------
    def run_simple_analysis(self):
        """
        Runs a staged pipeline analysis for the current view, unifying the execution path.
        """
        mode = self.setup_ctrl.current_pipeline_name()
        if not mode:
            QMessageBox.warning(
                self.window, "Analysis", "Please select a pipeline first."
            )
            return

        # Show which pipeline is selected
        self.window.statusBar().showMessage(f"Running {mode} analysis...", 2000)

        if mode.lower() != "sessile":
            # Fallback to old functional path for non-sessile modes
            image, roi_rect = self._get_analysis_image()
            if image is None:
                return
            try:
                modules = self._load_pipeline_modules(mode)
                helpers = self._prepare_helper_bundle(
                    mode, modules["helper_bundle_class"], roi_rect
                )
                self._run_analysis_and_update_ui(
                    mode, modules["analyze_func"], modules["draw_func"], image, helpers
                )
            except Exception as e:
                self.on_pipeline_error(f"An error occurred during analysis: {e}")
            return

        # Unified staged path for sessile
        image, roi_rect = self._get_analysis_image()
        if image is None:
            return

        try:
            # Prepare context with image and overlays
            from menipy.models.context import Context

            ctx = Context()
            ctx.image = image

            # Set substrate line if available
            substrate_line_item = self.preview_panel.substrate_line_item()
            if substrate_line_item:
                substrate_line = substrate_line_item.get_line_in_scene()
                if substrate_line:
                    p1 = substrate_line.p1() - roi_rect.topLeft()
                    p2 = substrate_line.p2() - roi_rect.topLeft()
                    ctx.substrate_line = ((p1.x(), p1.y()), (p2.x(), p2.y()))

            # Set scale and physics from calibration UI
            calibration_params = self.setup_ctrl.get_calibration_params()
            px_per_mm = (
                calibration_params["needle_length_mm"] / 100.0
            )  # Assume 100px needle for now
            ctx.scale = {"px_per_mm": px_per_mm}

            # Set edge detection settings
            if self.edge_detection_ctrl:
                ctx.edge_detection_settings = self.edge_detection_ctrl.settings

            # Set physics parameters from calibration
            calibration_params = self.setup_ctrl.get_calibration_params()
            ctx.physics = {
                "rho1": calibration_params["drop_density_kg_m3"],  # Drop density
                "rho2": calibration_params["fluid_density_kg_m3"],  # Fluid density
                "g": 9.80665,  # Gravity
            }

            # Run staged pipeline
            from menipy.pipelines.discover import PIPELINE_MAP

            pipeline_cls = PIPELINE_MAP.get("sessile")
            if not pipeline_cls:
                raise ValueError("Sessile pipeline not found")

            pipeline = pipeline_cls()
            # Prepare measurement tracking
            ctx_dict = self._prepare_measurement_tracking(**ctx.__dict__)
            ctx = pipeline.run(**ctx.__dict__)

            # Update UI with results
            if ctx.preview is not None:
                self.preview_panel.display(ctx.preview)
            if ctx.results:

                # Add to measurement history
                self.add_measurement_to_history(ctx, mode)
            self.window.statusBar().showMessage("Analysis complete.", 3000)

        except Exception as e:
            self.on_pipeline_error(f"An error occurred during staged analysis: {e}")

    def run_full(self) -> None:
        params = self.setup_ctrl.gather_run_params()
        name = (params.get("name") or "sessile" or "").lower()
        pipeline_cls = self.pipeline_map.get(name)
        if not pipeline_cls:
            QMessageBox.warning(self.window, "Run", f"Unknown pipeline: {name}")
            return

        ready, overlays = self._collect_acquisition_inputs()
        if not ready:
            return

        # Collect settings from dedicated controllers
        overlays.update(self._preprocessing_payload())
        overlays.update(self._edge_detection_payload())

        image = params.get("image")
        cam_id = params.get("cam_id")
        frames = params.get("frames")

        self.window.statusBar().showMessage(f"Running {name}.")

        run_kwargs = dict(overlays)
        if image is not None:
            run_kwargs["image"] = image
        if cam_id is not None:
            run_kwargs["camera"] = cam_id
        if frames is not None:
            run_kwargs["frames"] = frames

        if self.run_vm:
            try:
                run_kwargs_vm = {"pipeline": name, **run_kwargs}
                self.run_vm.run(**run_kwargs_vm)
                return
            except Exception as exc:
                print("[run_vm] fallback to direct run:", exc)

        self._run_pipeline_direct(pipeline_cls, **run_kwargs)

    def run_all(self) -> None:
        if not self.sops:
            self.run_full()
            return

        params = self.setup_ctrl.gather_run_params()
        name = (params.get("name") or "sessile" or "").lower()
        image = params.get("image")
        cam_id = params.get("cam_id")
        frames = params.get("frames")

        stages = self.setup_ctrl.collect_included_stages()
        if not stages:
            QMessageBox.warning(
                self.window, "Run All", "No stages enabled in the current SOP."
            )
            return

        overlays: Dict[str, Any] = {}
        if self._should_check_acquisition([stage.lower() for stage in stages]):
            ready, overlays = self._collect_acquisition_inputs()
            if not ready:
                return
        else:
            overlays = {}

        overlays.update(self._preprocessing_payload())
        overlays.update(self._edge_detection_payload())

        run_kwargs = dict(overlays)
        if image is not None:
            run_kwargs["image"] = image
        if cam_id is not None:
            run_kwargs["camera"] = cam_id
        if frames is not None:
            run_kwargs["frames"] = frames
        if params.get("calibration_params"):
            run_kwargs["calibration_params"] = params.get("calibration_params")

        run_kwargs["only"] = [stage for stage in stages]

        if self.run_vm and hasattr(self.run_vm, "run_subset"):
            try:
                self.window.statusBar().showMessage(f"Running {name} (SOP) .")
                run_kwargs_vm = dict(run_kwargs)
                run_kwargs_vm.pop("only", None)
                self.run_vm.run_subset(name, only=stages, **run_kwargs_vm)
                return
            except Exception as exc:
                print("[run_vm subset] falling back to full run:", exc)

        pipeline_cls = self.pipeline_map.get(name)
        if not pipeline_cls:
            QMessageBox.warning(self.window, "Run", f"Unknown pipeline: {name}")
            return

        self._run_pipeline_direct(pipeline_cls, **run_kwargs)

    def run_sop(self) -> None:
        """Run the current SOP (Standard Operating Procedure)."""
        if not self.sops:
            QMessageBox.warning(self.window, "Run SOP", "No SOP system available.")
            return

        # This would integrate with the SOP system
        # For now, delegate to run_all
        self.run_all()

    def run_stage(self, stage_name: str) -> None:
        stage_lower = (stage_name or "").strip().lower()
        params = self.setup_ctrl.gather_run_params()

        overlays: Dict[str, Any] = {}
        if stage_lower == "acquisition":
            ready, overlays = self._collect_acquisition_inputs()
            if not ready:
                return

        if self.run_vm and hasattr(self.run_vm, "run_subset"):
            try:
                run_kwargs_vm = {
                    "pipeline": params.get("name"),
                    "only": [stage_name],
                    "image": params.get("image"),
                    "camera": params.get("cam_id"),
                    "frames": params.get("frames"),
                    "roi": overlays.get("roi"),
                    "needle_rect": overlays.get("needle_rect"),
                    "contact_line": overlays.get("contact_line"),
                    "calibration_params": params.get("calibration_params"),
                }
                if self.preprocessing_ctrl:
                    run_kwargs_vm["preprocessing_settings"] = (
                        self.preprocessing_ctrl.settings.model_copy(deep=True)
                    )
                if self.edge_detection_ctrl:
                    # This was missing the call to the payload helper
                    run_kwargs_vm["edge_detection_settings"] = (
                        self.edge_detection_ctrl.settings.model_copy(deep=True)
                    )

                self.run_vm.run_subset(**run_kwargs_vm)
                return
            except Exception as exc:
                print("[run_vm single step] falling back to pipeline:", exc)
                # Continue to direct pipeline execution

        name = (params.get("name") or "sessile" or "").lower()
        pipeline_cls = self.pipeline_map.get(name)
        if not pipeline_cls:
            QMessageBox.warning(self.window, "Run", f"Unknown pipeline: {name}")
            return
        pipe = pipeline_cls()

        plan = pipe.build_plan(only=[stage_name], include_prereqs=True)
        if self._should_check_acquisition([n for n, _ in plan]):
            ready, overlays = self._collect_acquisition_inputs()
            if not ready:
                return

        try:
            run_kwargs_pipe = {
                "only": [stage_name],
                "include_prereqs": True,
                "image": params.get("image"),
                "camera": params.get("cam_id"),
                "frames": params.get("frames"),
                "roi": overlays.get("roi"),
                "needle_rect": overlays.get("needle_rect"),
                "contact_line": overlays.get("contact_line"),
                "calibration_params": params.get("calibration_params"),
            }
            if self.preprocessing_ctrl:
                run_kwargs_pipe["preprocessing_settings"] = (
                    self.preprocessing_ctrl.settings
                )
            if self.edge_detection_ctrl:
                run_kwargs_pipe["edge_detection_settings"] = (
                    self.edge_detection_ctrl.settings
                )

            ctx = pipe.run_with_plan(**run_kwargs_pipe)
            if ctx.preview is not None:
                self.preview_panel.display(ctx.preview)
            if getattr(ctx, "results", None):

                # Add to measurement history
                pipeline_name = (params.get("name") or "sessile" or "").lower()
                self.add_measurement_to_history(ctx, pipeline_name)
        except Exception as exc:
            self.on_pipeline_error(str(exc))

    def on_preview_ready(self, payload: Any) -> None:
        try:
            self.preview_panel.display(payload)
        except Exception:
            pass
        self.window.statusBar().showMessage("Preview updated", 1000)

    def on_results_ready(self, results: Mapping[str, Any]) -> None:
        params = self.setup_ctrl.gather_run_params()
        pipeline_name = (params.get("name") or "unknown").lower()
        self.results_panel.update_single_measurement(
            results, pipeline_name=pipeline_name
        )
        self.window.statusBar().showMessage("Results ready", 1000)

    def _prepare_measurement_tracking(self, **kwargs) -> dict:
        """Prepare measurement tracking fields for Context before pipeline runs."""
        from datetime import datetime
        from menipy.models.results import get_results_history

        history = get_results_history()
        sequence = len(history.measurements) + 1
        timestamp = datetime.now()
        measurement_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{sequence:03d}"

        # Add to run kwargs
        tracking_kwargs = dict(kwargs)
        tracking_kwargs["measurement_id"] = measurement_id
        tracking_kwargs["measurement_sequence"] = sequence
        return tracking_kwargs

    def add_measurement_to_history(self, ctx: Any, pipeline_name: str) -> None:
        """Add a completed measurement to the results history."""
        from datetime import datetime
        from menipy.models.results import MeasurementResult, get_results_history
        import uuid

        if not hasattr(ctx, "results") or not ctx.results:
            return

        # Generate measurement ID
        timestamp = datetime.now()
        sequence = len(get_results_history().measurements) + 1
        measurement_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{sequence:03d}"

        # Extract file information
        file_path = getattr(ctx, "image_path", None)
        file_name = None
        if file_path:
            from pathlib import Path

            file_name = Path(file_path).name
        elif hasattr(ctx, "image") and isinstance(ctx.image, str):
            from pathlib import Path

            file_name = Path(ctx.image).name

        # Create measurement result
        measurement = MeasurementResult(
            id=measurement_id,
            timestamp=timestamp,
            pipeline=pipeline_name,
            file_path=file_path,
            file_name=file_name,
            results=dict(ctx.results),
        )

        # Add to history
        self.results_panel.add_measurement(measurement)
        # Update status bar with measurement count
        total_measurements = len(get_results_history().measurements)
        self.window.statusBar().showMessage(
            f"Analysis complete - {total_measurements} measurements recorded", 3000
        )

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
        QMessageBox.critical(self.window, "Pipeline Error", message)
        self.window.statusBar().showMessage("Error", 1500)

    def _run_pipeline_direct(self, pipeline_cls: type, **kwargs: Any) -> None:
        try:
            pipeline = pipeline_cls()
            # Prepare measurement tracking
            kwargs = self._prepare_measurement_tracking(**kwargs)
            only = kwargs.pop("only", None)
            if only:
                ctx = pipeline.run_with_plan(only=only, include_prereqs=True, **kwargs)
            else:
                ctx = pipeline.run(**kwargs)
            if ctx.preview is not None:
                self.preview_panel.display(ctx.preview)
            if getattr(ctx, "results", None):

                # Add to measurement history
                params = self.setup_ctrl.gather_run_params()
                pipeline_name = (params.get("name") or "sessile" or "").lower()
                self.add_measurement_to_history(ctx, pipeline_name)
            self.window.statusBar().showMessage("Done", 1500)
        except Exception as exc:
            self.on_pipeline_error(str(exc))
