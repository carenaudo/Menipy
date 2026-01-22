"""
Main window controller coordinating GUI components.
"""

# src/menipy/gui/controllers/main_controller.py
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)
from PySide6.QtCore import QObject, Slot, QPointF
from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtGui import QImage, QColor

from menipy.gui.controllers.camera_manager import CameraManager
from menipy.gui.controllers.layout_manager import LayoutManager
from menipy.gui.controllers.dialog_coordinator import DialogCoordinator
from menipy.gui.controllers.dialog_coordinator import DialogCoordinator
from menipy.gui.controllers.overlay_manager import OverlayManager
from menipy.gui.controllers.image_manager import ImageManager

if TYPE_CHECKING:
    from menipy.gui.main_window import MainWindow
    from menipy.gui.controllers.pipeline_controller import PipelineController
    from menipy.gui.controllers.setup_panel_controller import SetupPanelController
    from menipy.gui.panels.preview_panel import PreviewPanel
    from menipy.gui.panels.results_panel import ResultsPanel
    from menipy.gui.services.camera_service import CameraController


class MainController(QObject):
    """Orchestrates the main application logic."""

    def __init__(self, window: MainWindow):
        super().__init__(window)
        self.window: MainWindow = window
        self.settings = window.settings
        self.setup_ctrl: SetupPanelController = window.setup_panel_ctrl
        self.preview_panel: PreviewPanel = window.preview_panel
        self.pipeline_ctrl: PipelineController = window.pipeline_ctrl
        self.preprocessing_ctrl = getattr(window, "preprocessing_ctrl", None)
        self.edge_detection_ctrl = getattr(window, "edge_detection_ctrl", None)
        self.camera_ctrl: CameraController | None = getattr(window, "camera_ctrl", None)
        self.results_panel: ResultsPanel | None = getattr(
            window, "results_panel_ctrl", None
        )
        self.results_panel: ResultsPanel | None = getattr(
            window, "results_panel_ctrl", None
        )

        # Initialize managers
        self.camera_manager = CameraManager(
            camera_ctrl=self.camera_ctrl,
            setup_ctrl=self.setup_ctrl,
            preview_panel=self.preview_panel,
            window=self.window,
            parent=self,
        )

        self.image_manager = ImageManager(
            window=self.window,
            setup_ctrl=self.setup_ctrl,
            preview_panel=self.preview_panel,
            preprocessing_ctrl=self.preprocessing_ctrl,
            parent=self
        )

        self.layout_manager = LayoutManager(window=self.window, settings=self.settings)

        self.dialog_coordinator = DialogCoordinator(
            window=self.window,
            settings=self.settings,
            preprocessing_ctrl=self.preprocessing_ctrl,
            edge_detection_ctrl=self.edge_detection_ctrl,
            image_loader=self.image_manager.load_preprocessing_image,
            parent=self,
        )

        self.overlay_manager = OverlayManager(
            image_view=getattr(self.preview_panel, 'image_view', None),
            settings=self.settings
        )

        self._wire_signals()
        if self.preview_panel:
            self.preview_panel.set_roi_callback(self._on_roi_selected)
            self.preview_panel.set_line_callback(self._on_contact_line_drawn)

        # Listen for edge-detection previews to show overlays on the main preview panel
        try:
            if self.edge_detection_ctrl and hasattr(
                self.edge_detection_ctrl, "previewRequested"
            ):
                self.edge_detection_ctrl.previewRequested.connect(
                    self._on_edge_detection_preview_image
                )
        except Exception:
            logger.debug(
                "Could not connect edge detection preview to main preview",
                exc_info=True,
            )

        self.camera_manager.on_source_mode_changed(self.setup_ctrl.current_mode())
        logger.info("MainController initialized and signals wired.")

    def _wire_signals(self):
        """Connect signals from child controllers to main controller slots."""
        # Setup Panel
        self.setup_ctrl.browse_requested.connect(self.image_manager.browse_image)
        self.setup_ctrl.browse_batch_requested.connect(self.image_manager.browse_batch_folder)
        self.setup_ctrl.preview_requested.connect(self.on_preview_requested)
        self.setup_ctrl.run_all_requested.connect(self.run_full_pipeline)
        self.setup_ctrl.play_stage_requested.connect(self.pipeline_ctrl.run_stage)
        self.setup_ctrl.analyze_requested.connect(self.analyze_current_view)
        self.setup_ctrl.config_stage_requested.connect(
            self.dialog_coordinator.show_dialog_for_stage
        )
        self.setup_ctrl.pipeline_changed.connect(self.on_pipeline_changed)
        self.setup_ctrl.auto_calibrate_requested.connect(self.on_auto_calibrate_requested)

        # Preview Panel
        self.setup_ctrl.draw_mode_requested.connect(self.preview_panel.set_draw_mode)
        self.setup_ctrl.clear_overlays_requested.connect(
            self.preview_panel.clear_overlays
        )
        self.setup_ctrl.source_mode_changed.connect(
            self.camera_manager.on_source_mode_changed
        )

        # Pipeline Runner VM
        run_vm = getattr(self.window, "run_vm", None)
        if run_vm:
            if hasattr(run_vm, "preview_ready"):
                run_vm.preview_ready.connect(self.pipeline_ctrl.on_preview_ready)
            if hasattr(run_vm, "results_ready"):
                run_vm.results_ready.connect(self.pipeline_ctrl.on_results_ready)
            if hasattr(run_vm, "logs_ready"):
                run_vm.logs_ready.connect(self.pipeline_ctrl.append_logs)
            if hasattr(run_vm, "error"):
                run_vm.error.connect(self.pipeline_ctrl.on_pipeline_error)
            elif hasattr(run_vm, "error_occurred"):
                run_vm.error_occurred.connect(self.pipeline_ctrl.on_pipeline_error)

    @Slot()
    def analyze_current_view(self):
        """Triggers a simple, direct analysis of the current view."""
        self.pipeline_ctrl.run_simple_analysis()

    @Slot()
    def browse_image(self):
        """Delegates image browsing to the ImageManager."""
        self.image_manager.browse_image()

    @Slot(bool)
    def select_camera(self, on: bool):
        """Switches the input mode to camera."""
        self.setup_ctrl.set_camera_enabled(on)
        if not on and self.camera_ctrl:
            self.camera_ctrl.stop()

    @Slot()
    def run_full_pipeline(self):
        """Triggers a full pipeline run."""
        self.pipeline_ctrl.run_full()

    @Slot()
    def on_preview_requested(self) -> None:
        """Loads the currently selected source into the preview panel."""
        params = self.setup_ctrl.gather_run_params()
        image_path = params.get("image")
        if image_path:
            try:
                self.preview_panel.load_path(image_path)
                self.window.statusBar().showMessage("Preview loaded", 1500)
            except Exception as exc:
                logger.error(f"Failed to load preview: {exc}")
                QMessageBox.warning(
                    self.window, "Preview", f"Could not load preview.\n{exc}"
                )
            return

        batch_folder = params.get("batch_folder")
        if batch_folder:
            try:
                folder = Path(batch_folder)
                extensions = getattr(self.setup_ctrl, "_IMAGE_EXTENSIONS", None)
                for candidate in sorted(folder.iterdir()):
                    if not candidate.is_file():
                        continue
                    if extensions and candidate.suffix.lower() not in extensions:
                        continue
                    self.preview_panel.load_path(str(candidate))
                    self.window.statusBar().showMessage("Batch preview loaded", 1500)
                    return
            except Exception as exc:
                logger.error(f"Failed to load batch preview: {exc}")
                QMessageBox.warning(
                    self.window, "Preview", f"Could not load batch preview.\n{exc}"
                )
                return

        self.window.statusBar().showMessage("No preview source available", 2000)

    @Slot()
    def on_auto_calibrate_requested(self) -> None:
        """Launch auto-calibration wizard."""
        image = self.image_manager.load_preprocessing_image()
        if image is None:
            self.window.statusBar().showMessage('No image loaded for calibration', 2000)
            QMessageBox.warning(
                self.window,
                'Auto-Calibrate',
                'Please load an image before running auto-calibration.'
            )
            return
        
        pipeline_name = self.setup_ctrl.current_pipeline_name() or 'sessile'
        
        try:
            from menipy.gui.dialogs.calibration_wizard_dialog import CalibrationWizardDialog
            
            wizard = CalibrationWizardDialog(image, pipeline_name, self.window)
            wizard.calibration_complete.connect(self._on_calibration_complete)
            wizard.exec()
        except Exception as exc:
            logger.exception('Failed to open calibration wizard')
            QMessageBox.critical(
                self.window,
                'Auto-Calibrate Error',
                f'Failed to open calibration wizard:\n{exc}'
            )
    
    @Slot(object)
    def _on_calibration_complete(self, result) -> None:
        """Apply calibration results to preview and settings."""
        try:
            from menipy.common.auto_calibrator import CalibrationResult
            if not isinstance(result, CalibrationResult):
                return
            
            # Delegate drawing to OverlayManager
            self.overlay_manager.draw_calibration_result(result)
            
            # Update controllers with calibration data
            if result.roi_rect and self.preprocessing_ctrl:
                self.preprocessing_ctrl.update_geometry(roi=result.roi_rect)
                logger.info(f'Calibration: ROI set to {result.roi_rect}')
            
            if result.substrate_line and self.preprocessing_ctrl:
                self.preprocessing_ctrl.update_geometry(contact_line=result.substrate_line)
            
            # Report success
            conf = result.confidence_scores.get('overall', 0.0)
            self.window.statusBar().showMessage(
                f'Calibration complete (confidence: {conf*100:.0f}%)', 3000
            )
            logger.info(f'Calibration applied: ROI={result.roi_rect}, needle={result.needle_rect}, substrate={result.substrate_line is not None}')
            
        except Exception as exc:
            logger.exception('Failed to apply calibration results')
            self.window.statusBar().showMessage('Failed to apply calibration', 2000)

    @Slot()
    def stop_pipeline(self):
        """Stops any active pipeline run."""
        if self.window.runner:
            self.window.runner.pool.clear()
            logger.info("Cleared pending tasks in the thread pool.")
            self.window.statusBar().showMessage("Stop request sent.", 2000)

    @Slot(str)
    def on_pipeline_changed(self, pipeline_name: str):
        """Saves the selected pipeline to settings."""
        self.settings.selected_pipeline = pipeline_name

    @Slot()
    def show_about_dialog(self):
        """Displays the application's About box."""
        QMessageBox.about(
            self.window,
            "About Menipy",
            "<b>Menipy Droplet Analysis</b>"
            "<p>A toolkit for analyzing droplet shapes from images.</p>"
            "<p>This software is under active development.</p>",
        )

    @Slot(object)
    def _on_roi_selected(self, rect) -> None:
        if rect is None or self.preprocessing_ctrl is None:
            return
        try:
            left = int(round(rect.left()))
            top = int(round(rect.top()))
            width = int(round(rect.width()))
            height = int(round(rect.height()))
        except AttributeError:
            return
        if width <= 0 or height <= 0:
            return
        roi = (left, top, width, height)
        image = self.image_manager.load_preprocessing_image()
        if image is None:
            logger.debug("Preprocessing: could not load source image for ROI update.")
            return
        contact_line = (
            self.preview_panel.contact_line_segment()
            if hasattr(self.preview_panel, "contact_line_segment")
            and self.preview_panel.contact_line_segment()
            else None
        )
        if not self.preprocessing_ctrl.has_source():
            self.preprocessing_ctrl.set_source(
                image, roi=roi, contact_line=contact_line
            )
        else:
            self.preprocessing_ctrl.update_geometry(roi=roi, contact_line=contact_line)
        self.preprocessing_ctrl.run()

    @Slot(object)
    def _on_contact_line_drawn(self, line) -> None:
        if self.preprocessing_ctrl is None or line is None:
            return
        try:
            p1 = line.p1()
            p2 = line.p2()
        except AttributeError:
            return
        contact = (
            (int(round(p1.x())), int(round(p1.y()))),
            (int(round(p2.x())), int(round(p2.y()))),
        )
        self.preprocessing_ctrl.update_geometry(contact_line=contact)
        if self.preprocessing_ctrl.has_source():
            self.preprocessing_ctrl.run()
        # Forward contact line to edge detection controller
        try:
            if self.edge_detection_ctrl is not None:
                self.edge_detection_ctrl.set_contact_line(contact)
        except Exception:
            logger.debug(
                "Failed to forward contact line to edge detection controller",
                exc_info=True,
            )

    @Slot(object, dict)
    def _on_edge_detection_preview_image(self, image, metadata: dict) -> None:
        """Receive preview image + metadata from edge detection controller and display it."""
        try:
            if not self.preview_panel:
                return
            try:
                self.preview_panel.display(image)
            except Exception:
                self.preview_panel.display(image)

            if (
                isinstance(metadata, dict)
                and self.preview_panel
                and getattr(self.preview_panel, "image_view", None)
            ):
                # Delegate overlay drawing to OverlayManager
                self.overlay_manager.draw_edge_detection_preview(metadata)
            
        except Exception:
            logger.debug(
                "Error while handling edge detection preview image", exc_info=True
            )

    def shutdown(self):
        """Saves settings and performs cleanup before the application closes."""
        logger.info("MainController shutting down...")
        self.layout_manager.save_layout()
        self.camera_manager.shutdown()
