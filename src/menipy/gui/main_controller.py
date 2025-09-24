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
from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QFileDialog, QMessageBox, QDialog
from PySide6.QtGui import QImage

from menipy.gui.services.camera_service import CameraConfig
from menipy.gui.dialogs.acquisition_config_dialog import AcquisitionConfigDialog
from menipy.gui.dialogs.preprocessing_config_dialog import PreprocessingConfigDialog
if TYPE_CHECKING:
    from menipy.gui.mainwindow import MainWindow
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
        self.camera_ctrl: CameraController | None = getattr(window, 'camera_ctrl', None)
        self.results_panel: ResultsPanel | None = getattr(window, 'results_panel_ctrl', None)
        self._camera_preview_logged = False
        self._active_camera_device: int | None = None
        self._cached_image_path: Optional[str] = None
        self._cached_image_data: Optional[np.ndarray] = None

        if self.camera_ctrl:
            try:
                self.camera_ctrl.frame_ready.connect(self._on_camera_frame)
            except Exception as exc:
                logger.warning(f'Could not connect camera frame signal: {exc}')
            if hasattr(self.camera_ctrl, 'error'):
                try:
                    self.camera_ctrl.error.connect(self._on_camera_error)
                except Exception:
                    pass

        self._wire_signals()
        if self.preview_panel:
            self.preview_panel.set_roi_callback(self._on_roi_selected)
            self.preview_panel.set_line_callback(self._on_contact_line_drawn)
        self.on_source_mode_changed(self.setup_ctrl.current_mode())
        logger.info("MainController initialized and signals wired.")

    def _wire_signals(self):
        """Connect signals from child controllers to main controller slots."""
        # Setup Panel
        self.setup_ctrl.browse_requested.connect(self.browse_image)
        self.setup_ctrl.browse_batch_requested.connect(self.browse_batch_folder)
        self.setup_ctrl.preview_requested.connect(self.on_preview_requested)
        self.setup_ctrl.run_all_requested.connect(self.run_full_pipeline)
        self.setup_ctrl.play_stage_requested.connect(self.pipeline_ctrl.run_stage)
        self.setup_ctrl.config_stage_requested.connect(self.on_config_stage_requested)
        self.setup_ctrl.pipeline_changed.connect(self.on_pipeline_changed)

        # Preview Panel
        self.setup_ctrl.draw_mode_requested.connect(self.preview_panel.set_draw_mode)
        self.setup_ctrl.clear_overlays_requested.connect(self.preview_panel.clear_overlays)
        self.setup_ctrl.source_mode_changed.connect(self.on_source_mode_changed)

        # Pipeline Runner VM
        run_vm = getattr(self.window, 'run_vm', None)
        if run_vm:
            if hasattr(run_vm, 'preview_ready'):
                run_vm.preview_ready.connect(self.pipeline_ctrl.on_preview_ready)
            if hasattr(run_vm, 'results_ready'):
                run_vm.results_ready.connect(self.pipeline_ctrl.on_results_ready)
            if hasattr(run_vm, 'logs_ready'):
                run_vm.logs_ready.connect(self.pipeline_ctrl.append_logs)
            if hasattr(run_vm, 'error'):
                run_vm.error.connect(self.pipeline_ctrl.on_pipeline_error)
            elif hasattr(run_vm, 'error_occurred'):
                run_vm.error_occurred.connect(self.pipeline_ctrl.on_pipeline_error)

    @Slot()
    def browse_image(self):
        """Opens a file dialog to select a single image."""
        start_dir = str(Path(self.settings.last_image_path or Path.home()).parent)
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Open Image", start_dir, "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self.settings.last_image_path = path
            self.setup_ctrl.set_image_path(path)
            self.preview_panel.load_path(path)
            if self.preprocessing_ctrl is not None:
                image = self._load_preprocessing_image(path_override=path)
                if image is not None:
                    self.preprocessing_ctrl.set_source(image)
                    try:
                        self.preprocessing_ctrl.run()
                    except Exception as exc:  # pragma: no cover - guard runtime
                        logger.debug('Initial preprocessing run failed: %s', exc)

    @Slot()
    def browse_batch_folder(self):
        """Opens a file dialog to select a batch processing folder."""
        start_dir = self.setup_ctrl.batch_path() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self.window, "Select Batch Folder", start_dir)
        if path:
            self.setup_ctrl.set_batch_path(path)

    @Slot(bool)
    def select_camera(self, on: bool):
        """Switches the input mode to camera."""
        self.setup_ctrl.set_camera_enabled(on)
        if not on and self.camera_ctrl:
            self.camera_ctrl.stop()



    @Slot(str)
    def on_config_stage_requested(self, stage_name: str) -> None:
        stage = (stage_name or "").strip().lower()
        if stage == "preprocessing":
            if not self.preprocessing_ctrl:
                QMessageBox.information(self.window, "Preprocessing", "Preprocessing controller is not available.")
                return
            dialog = PreprocessingConfigDialog(self.preprocessing_ctrl.settings, parent=self.window)
            dialog.previewRequested.connect(self._on_preprocessing_preview)
            if dialog.exec() == QDialog.Accepted:
                self.preprocessing_ctrl.set_settings(dialog.settings())
                try:
                    self.preprocessing_ctrl.run()
                except Exception as exc:  # pragma: no cover - runtime guard
                    logger.warning("Preprocessing preview failed: %s", exc)
            else:
                logger.info("Preprocessing configuration cancelled")
            return

        if stage != "acquisition":
            QMessageBox.information(
                self.window,
                "Stage Configuration",
                f"Configuration for '{stage_name}' is not available yet.",
            )
            return

        dialog = AcquisitionConfigDialog(
            contact_line_required=getattr(self.settings, "acquisition_requires_contact_line", False),
            parent=self.window,
        )
        if dialog.exec() == QDialog.Accepted:
            requires_contact_line = dialog.contact_line_required
            if getattr(self.settings, "acquisition_requires_contact_line", False) != requires_contact_line:
                self.settings.acquisition_requires_contact_line = requires_contact_line
                try:
                    self.settings.save()
                except Exception as exc:
                    logger.warning("Failed to persist acquisition settings: %s", exc)
                logger.info(
                    "Acquisition configuration updated: contact line required=%s",
                    requires_contact_line,
                )
        else:
            logger.info("Acquisition configuration cancelled")


    @Slot()
    def run_full_pipeline(self):
        """Triggers a full pipeline run."""
        self.pipeline_ctrl.run_full()

    @Slot(object)
    def _on_camera_frame(self, frame: object) -> None:
        """Render live camera frames in the preview panel."""
        if not self._camera_preview_logged:
            logger.info("Camera preview streaming from device %s", self._active_camera_device if self._active_camera_device is not None else 'unknown')
            self._camera_preview_logged = True
        try:
            self.preview_panel.display(frame)
        except Exception as exc:
            logger.error(f'Failed to display camera frame: {exc}')

    @Slot(str)
    def _on_camera_error(self, message: str) -> None:
        self.window.statusBar().showMessage(message, 2000)
        logger.error(message)

    @Slot()
    def on_preview_requested(self) -> None:
        """Loads the currently selected source into the preview panel."""
        params = self.setup_ctrl.gather_run_params()
        image_path = params.get('image')
        if image_path:
            try:
                self.preview_panel.load_path(image_path)
                self.window.statusBar().showMessage('Preview loaded', 1500)
            except Exception as exc:
                logger.error(f'Failed to load preview: {exc}')
                QMessageBox.warning(self.window, 'Preview', f'Could not load preview.\n{exc}')
            return

        batch_folder = params.get('batch_folder')
        if batch_folder:
            try:
                folder = Path(batch_folder)
                extensions = getattr(self.setup_ctrl, '_IMAGE_EXTENSIONS', None)
                for candidate in sorted(folder.iterdir()):
                    if not candidate.is_file():
                        continue
                    if extensions and candidate.suffix.lower() not in extensions:
                        continue
                    self.preview_panel.load_path(str(candidate))
                    self.window.statusBar().showMessage('Batch preview loaded', 1500)
                    return
            except Exception as exc:
                logger.error(f'Failed to load batch preview: {exc}')
                QMessageBox.warning(self.window, 'Preview', f'Could not load batch preview.\n{exc}')
                return

        self.window.statusBar().showMessage('No preview source available', 2000)

    @Slot()
    def stop_pipeline(self):
        """Stops any active pipeline run."""
        if self.window.runner:
            # The runner uses a global thread pool, we can't easily stop a specific job
            # For now, we can clear pending tasks. A more robust solution would
            # involve cancellable QRunnables.
            self.window.runner.pool.clear()
            logger.info("Cleared pending tasks in the thread pool.")
            self.window.statusBar().showMessage("Stop request sent.", 2000)

    @Slot(str)
    def on_pipeline_changed(self, pipeline_name: str):
        """Saves the selected pipeline to settings."""
        self.settings.selected_pipeline = pipeline_name

    @Slot(str)
    def on_source_mode_changed(self, mode: str) -> None:
        """Starts or stops the camera service based on the active source mode."""
        camera_ctrl: CameraController | None = getattr(self, 'camera_ctrl', None)
        if not camera_ctrl:
            return

        if mode == getattr(self.setup_ctrl, 'MODE_CAMERA', 'camera'):
            params = self.setup_ctrl.gather_run_params()
            cam_id = params.get('cam_id')
            try:
                device = int(cam_id) if cam_id is not None else 0
            except (TypeError, ValueError):
                device = 0
            config = CameraConfig(device=device)
            try:
                self._active_camera_device = device
                self._camera_preview_logged = False
                logger.info("Starting camera preview on device %s", device)
                camera_ctrl.start(config)
                self.window.statusBar().showMessage(f"Camera {device} streaming", 1500)
            except Exception as exc:
                self._active_camera_device = None
                logger.error(f"Failed to start camera {device}: {exc}")
                QMessageBox.warning(self.window, 'Camera', f"Could not start camera {device}.\n{exc}")
        else:
            camera_ctrl.stop()
            logger.info("Camera preview stopped")
            self._active_camera_device = None
            self._camera_preview_logged = False

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
        image = self._load_preprocessing_image()
        if image is None:
            logger.debug('Preprocessing: could not load source image for ROI update.')
            return
        contact_line = self.preview_panel.contact_line_segment() if hasattr(self.preview_panel, 'contact_line_segment') else None
        if not self.preprocessing_ctrl.has_source():
            self.preprocessing_ctrl.set_source(image, roi=roi, contact_line=contact_line)
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
        contact = ((int(round(p1.x())), int(round(p1.y()))), (int(round(p2.x())), int(round(p2.y()))))
        self.preprocessing_ctrl.update_geometry(contact_line=contact)
        if self.preprocessing_ctrl.has_source():
            self.preprocessing_ctrl.run()

    def _on_preprocessing_preview(self, settings) -> None:
        if not self.preprocessing_ctrl:
            return
        self.preprocessing_ctrl.set_settings(settings)
        if not self.preprocessing_ctrl.has_source():
            image = self._load_preprocessing_image()
            if image is None:
                return
            self.preprocessing_ctrl.set_source(image)
        try:
            self.preprocessing_ctrl.run()
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Preprocessing preview failed: %s", exc)

    def _load_preprocessing_image(self, path_override: Optional[str] = None) -> Optional[np.ndarray]:
        if self.preprocessing_ctrl is None:
            return None
        path = path_override or self.setup_ctrl.image_path()
        if path:
            if path == self._cached_image_path and self._cached_image_data is not None:
                return self._cached_image_data.copy()
            img = None
            if cv2 is not None:
                try:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                except Exception as exc:  # pragma: no cover - guard optional dependency
                    logger.debug('cv2.imread failed for %s: %s', path, exc)
                    img = None
            if img is None:
                qimg = QImage(path)
                if qimg.isNull():
                    logger.warning('Unable to load image for preprocessing: %s', path)
                    return None
                img = self._qimage_to_bgr(qimg)
            self._cached_image_path = path
            self._cached_image_data = img
            return img.copy()
        if self._cached_image_data is not None:
            return self._cached_image_data.copy()
        return None

    def _qimage_to_bgr(self, qimg: QImage) -> np.ndarray:
        converted = qimg.convertToFormat(QImage.Format.Format_RGB888)
        width = converted.width()
        height = converted.height()
        ptr = converted.constBits()
        ptr.setsize(converted.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        return arr[..., ::-1].copy()

    def shutdown(self):
        """Saves settings and performs cleanup before the application closes."""
        logger.info("MainController shutting down...")
        try:
            self.settings.main_window_geom_b64 = self.window.saveGeometry().toBase64().data().decode("ascii")
            self.settings.main_window_state_b64 = self.window.saveState().toBase64().data().decode("ascii")
            self.settings.splitter_sizes = self.window.rootSplitter.sizes()
            self.settings.save()
            logger.info("Window state and settings saved.")
        except Exception as e:
            logger.error(f"Failed to save settings on shutdown: {e}")
        if self.camera_ctrl:
            self.camera_ctrl.shutdown()