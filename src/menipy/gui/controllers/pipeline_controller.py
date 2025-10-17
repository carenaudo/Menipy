"""Pipeline execution helper for Menipy GUI."""
from __future__ import annotations

import importlib
import logging
from typing import Any, Mapping, Optional, Dict

from menipy.gui.controllers.preprocessing_controller import PreprocessingPipelineController
from menipy.gui.controllers.edge_detection_controller import EdgeDetectionPipelineController
from PySide6.QtWidgets import QMessageBox, QPlainTextEdit, QMainWindow

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

        roi_rect = preview.roi_rect() if hasattr(preview, 'roi_rect') else None
        if not roi_rect:
            missing.append('ROI')
        else:
            overlays['roi'] = roi_rect

        needle_rect = preview.needle_rect() if hasattr(preview, 'needle_rect') else None
        if not needle_rect:
            missing.append('needle region')
        else:
            overlays['needle_rect'] = needle_rect

        requires_contact = bool(getattr(self.window.settings, 'acquisition_requires_contact_line', False))
        contact_line = preview.contact_line_segment() if hasattr(preview, 'contact_line_segment') else None
        if requires_contact:
            if not contact_line:
                missing.append('contact line')
            else:
                overlays['contact_line'] = contact_line
        elif contact_line:
            overlays['contact_line'] = contact_line

        if missing:
            QMessageBox.warning(
                self.window,
                'Acquisition Requirements',
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
            'preprocessing_settings': ctrl.settings.model_copy(deep=True),
        }
        try:
            payload['preprocessing_markers'] = ctrl.markers.model_copy(deep=True)
        except Exception:
            payload['preprocessing_markers'] = ctrl.markers
        return payload

    def _edge_detection_payload(self) -> Dict[str, Any]:
        ctrl = self.edge_detection_ctrl
        if not ctrl:
            return {}
        payload: Dict[str, Any] = {
            'edge_detection_settings': ctrl.settings.model_copy(deep=True),
        }
        return payload

    def _should_check_acquisition(self, stages: Optional[list[str]]) -> bool:
        if not stages:
            return True
        return any((stage or '').strip().lower() == 'acquisition' for stage in stages)

    def _get_analysis_image(self):
        """Validates and returns the cropped image and its ROI."""
        if self.preview_panel.image_item is None or self.preview_panel.image_item.pixmap().isNull():
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
        geometry_module = importlib.import_module(f".geometry", package=f"menipy.pipelines.{mode}")
        drawing_module = importlib.import_module(f".drawing", package=f"menipy.pipelines.{mode}")

        # The actual 'analyze' function needs to be implemented in the pipeline files.
        # For now, we assume it exists.
        analyze_func = getattr(geometry_module, "analyze", None)
        if analyze_func is None:
            raise AttributeError(f"'analyze' function not found in {geometry_module.__name__}")

        return {
            "analyze_func": analyze_func,
            "draw_func": getattr(drawing_module, f"draw_{mode}_overlay"),
            "helper_bundle_class": getattr(geometry_module, "HelperBundle"),
        }

    def _prepare_helper_bundle(self, mode: str, helper_bundle_class, roi_rect):
        """Prepares and returns the mode-specific HelperBundle."""
        # These attributes are on the main window's tabs
        px_per_mm = self.window.calibration_tab.get_scale()
        if mode == "pendant":
            needle_diam_mm = self.window.calibration_tab.get_needle_diameter()
            liquid_rho = self.window.calibration_tab.get_liquid_density()
            air_rho = self.window.calibration_tab.get_air_density()
            delta_rho = liquid_rho - air_rho
            # apex_window_px could be sourced from a new UI control in the future
            return helper_bundle_class(
                px_per_mm=px_per_mm,
                needle_diam_mm=needle_diam_mm,
                delta_rho=delta_rho,
                # g and apex_window_px will use their defaults from the dataclass
            )

        if mode == "sessile":
            substrate_line_item = self.preview_panel.substrate_line_item()
            substrate_line = substrate_line_item.get_line_in_scene() if substrate_line_item else None
            if substrate_line:
                p1 = substrate_line.p1() - roi_rect.topLeft()
                p2 = substrate_line.p2() - roi_rect.topLeft()
                substrate_line = ((p1.x(), p1.y()), (p2.x(), p2.y()))

            liquid_rho = self.window.calibration_tab.get_liquid_density()
            air_rho = self.window.calibration_tab.get_air_density()
            delta_rho = liquid_rho - air_rho
            # Assuming contact_points is managed by the controller
            contact_points = self.contact_points if self.contact_points else None
            return helper_bundle_class(
                px_per_mm=px_per_mm,
                substrate_line=substrate_line,
                contact_points=contact_points,
                delta_rho=delta_rho,
                # contact_point_tolerance_px will use its default from the dataclass.
                # A UI control could be added to override it.
            )

        QMessageBox.warning(self.window, "Unsupported Mode", f"Analysis mode '{mode}' is not supported.")
        return None

    def _run_analysis_and_update_ui(self, mode, analyze_func, draw_func, image, helpers):
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
            self.results_panel.update(metrics.derived)
            self.window.statusBar().showMessage("Analysis complete.", 3000)
        except Exception as e:
            self.on_pipeline_error(f"An error occurred during analysis: {e}")

    # ------------------------------------------------------------------
    # Slots wired by MainWindow
    # ------------------------------------------------------------------
    def run_simple_analysis(self):
        """
        Runs a direct analysis based on the currently selected pipeline mode,
        bypassing the full SOP runner.
        """
        mode = self.setup_ctrl.current_pipeline_name()
        if not mode:
            QMessageBox.warning(self.window, "Analysis", "Please select a pipeline first.")
            return

        image, roi_rect = self._get_analysis_image()
        if image is None:
            return

        try:
            modules = self._load_pipeline_modules(mode)
            helpers = self._prepare_helper_bundle(mode, modules["helper_bundle_class"], roi_rect)
            self._run_analysis_and_update_ui(mode, modules["analyze_func"], modules["draw_func"], image, helpers)
        except Exception as e:
            self.on_pipeline_error(f"An error occurred during analysis: {e}")

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
            run_kwargs['image'] = image
        if cam_id is not None:
            run_kwargs['camera'] = cam_id
        if frames is not None:
            run_kwargs['frames'] = frames

        if self.run_vm:
            try:
                run_kwargs_vm = {'pipeline': name, **run_kwargs}
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
            QMessageBox.warning(self.window, "Run All", "No stages enabled in the current SOP.")
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
            run_kwargs['image'] = image
        if cam_id is not None:
            run_kwargs['camera'] = cam_id
        if frames is not None:
            run_kwargs['frames'] = frames

        run_kwargs['only'] = [stage for stage in stages]

        if self.run_vm and hasattr(self.run_vm, "run_subset"):
            try:
                self.window.statusBar().showMessage(f"Running {name} (SOP) .")
                run_kwargs_vm = dict(run_kwargs)
                run_kwargs_vm.pop('only', None)
                self.run_vm.run_subset(name, only=stages, **run_kwargs_vm)
                return
            except Exception as exc:
                print("[run_vm subset] falling back to full run:", exc)

        pipeline_cls = self.pipeline_map.get(name)
        if not pipeline_cls:
            QMessageBox.warning(self.window, "Run", f"Unknown pipeline: {name}")
            return

        self._run_pipeline_direct(pipeline_cls, **run_kwargs)

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
                    "roi": overlays.get('roi'),
                    "needle_rect": overlays.get('needle_rect'),
                    "contact_line": overlays.get('contact_line'),
                }
                if self.preprocessing_ctrl:
                    run_kwargs_vm["preprocessing_settings"] = self.preprocessing_ctrl.settings.model_copy(deep=True)
                if self.edge_detection_ctrl:
                    # This was missing the call to the payload helper
                    run_kwargs_vm["edge_detection_settings"] = self.edge_detection_ctrl.settings.model_copy(deep=True)

                self.run_vm.run_subset(**run_kwargs_vm)
                return
            except Exception as exc:
                print("[run_vm single step] falling back to pipeline:", exc)

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
                "roi": overlays.get('roi'),
                "needle_rect": overlays.get('needle_rect'),
                "contact_line": overlays.get('contact_line'),
            }
            if self.preprocessing_ctrl:
                run_kwargs_pipe["preprocessing_settings"] = self.preprocessing_ctrl.settings
            if self.edge_detection_ctrl:
                run_kwargs_pipe["edge_detection_settings"] = self.edge_detection_ctrl.settings

            pipe.run_with_plan(**run_kwargs_pipe)
        except Exception as exc:
            self.on_pipeline_error(str(exc))

    def on_preview_ready(self, payload: Any) -> None:
        try:
            self.preview_panel.display(payload)
        except Exception:
            pass
        self.window.statusBar().showMessage("Preview updated", 1000)

    def on_results_ready(self, results: Mapping[str, Any]) -> None:
        self.results_panel.update(results)
        self.window.statusBar().showMessage("Results ready", 1000)

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
            only = kwargs.pop("only", None)
            if only:
                ctx = pipeline.run_with_plan(only=only, include_prereqs=True, **kwargs)
            else:
                ctx = pipeline.run(**kwargs)
            if getattr(ctx, "preview", None) is not None:
                self.preview_panel.display(ctx.preview)
            if getattr(ctx, "results", None):
                self.results_panel.update(ctx.results)
            self.window.statusBar().showMessage("Done", 1500)
        except Exception as exc:
            self.on_pipeline_error(str(exc))