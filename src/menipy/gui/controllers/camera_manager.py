"""
Camera management for the main application.

Handles camera streaming, frame handling, and device switching.
Extracted from MainController to adhere to Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, Signal, Slot

if TYPE_CHECKING:
    from menipy.gui.services.camera_service import CameraController, CameraConfig
    from menipy.gui.controllers.setup_panel_controller import SetupPanelController
    from menipy.gui.panels.preview_panel import PreviewPanel
    from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


class CameraManager(QObject):
    """Manages camera streaming and frame handling."""

    frame_displayed = Signal()  # Emitted after a frame is displayed
    error = Signal(str)  # Emitted on camera error

    def __init__(
        self,
        camera_ctrl: Optional[CameraController],
        setup_ctrl: SetupPanelController,
        preview_panel: PreviewPanel,
        window: QMainWindow,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.camera_ctrl = camera_ctrl
        self.setup_ctrl = setup_ctrl
        self.preview_panel = preview_panel
        self.window = window

        self._active_camera_device: Optional[int] = None
        self._camera_preview_logged = False

        # Wire camera signals
        if self.camera_ctrl:
            try:
                self.camera_ctrl.frame_ready.connect(self._on_camera_frame)
            except Exception as exc:
                logger.warning(f"Could not connect camera frame signal: {exc}")
            if hasattr(self.camera_ctrl, "error"):
                try:
                    self.camera_ctrl.error.connect(self._on_camera_error)
                except Exception:
                    pass

    @Slot(str)
    def on_source_mode_changed(self, mode: str) -> None:
        """Starts or stops the camera service based on the active source mode."""
        if not self.camera_ctrl:
            return

        MODE_CAMERA = getattr(self.setup_ctrl, "MODE_CAMERA", "camera")

        if mode == MODE_CAMERA:
            params = self.setup_ctrl.gather_run_params()
            cam_id = params.get("cam_id")
            try:
                device = int(cam_id) if cam_id is not None else 0
            except (TypeError, ValueError):
                device = 0

            # Import here to avoid circular imports
            from menipy.gui.services.camera_service import CameraConfig

            config = CameraConfig(device=device)

            try:
                self._active_camera_device = device
                self._camera_preview_logged = False
                logger.info("Starting camera preview on device %s", device)
                self.camera_ctrl.start(config)
                self.window.statusBar().showMessage(f"Camera {device} streaming", 1500)
            except Exception as exc:
                self._active_camera_device = None
                logger.error(f"Failed to start camera {device}: {exc}")
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self.window, "Camera", f"Could not start camera {device}.\n{exc}"
                )
        else:
            self.camera_ctrl.stop()
            logger.info("Camera preview stopped")
            self._active_camera_device = None
            self._camera_preview_logged = False

    @Slot(object)
    def _on_camera_frame(self, frame: object) -> None:
        """Render live camera frames in the preview panel."""
        if not self._camera_preview_logged:
            logger.info(
                "Camera preview streaming from device %s",
                (
                    self._active_camera_device
                    if self._active_camera_device is not None
                    else "unknown"
                ),
            )
            self._camera_preview_logged = True
        try:
            self.preview_panel.display(frame)
            self.frame_displayed.emit()
        except Exception as exc:
            logger.error(f"Failed to display camera frame: {exc}")

    @Slot(str)
    def _on_camera_error(self, message: str) -> None:
        """Handle camera errors."""
        self.window.statusBar().showMessage(message, 2000)
        logger.error(message)
        self.error.emit(message)

    def shutdown(self) -> None:
        """Shutdown the camera controller."""
        if self.camera_ctrl:
            self.camera_ctrl.shutdown()
