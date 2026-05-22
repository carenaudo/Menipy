"""Dialog for camera capture settings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
)

from menipy.gui.services.camera_service import (
    CameraDevice,
    default_camera_fps_values,
    default_camera_resolutions,
)


class CameraSettingsDialog(QDialog):
    """Edit camera selection and capture settings."""

    def __init__(
        self,
        *,
        cameras: Sequence[CameraDevice],
        current_device: int,
        frames: int,
        fps: int,
        width: int | None,
        height: int | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Camera Settings")
        self.setObjectName("cameraSettingsDialog")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.cameraCombo = QComboBox(self)
        self.cameraCombo.setObjectName("cameraSettingsCameraCombo")
        for camera in cameras or [CameraDevice(0, "Camera 0")]:
            self.cameraCombo.addItem(camera.label, camera.device)
        camera_index = self.cameraCombo.findData(int(current_device))
        self.cameraCombo.setCurrentIndex(max(0, camera_index))
        form.addRow("Camera", self.cameraCombo)

        self.framesSpin = QSpinBox(self)
        self.framesSpin.setObjectName("cameraSettingsFramesSpin")
        self.framesSpin.setRange(1, 9999)
        self.framesSpin.setValue(max(1, int(frames or 1)))
        form.addRow("Frames to capture", self.framesSpin)

        self.fpsCombo = QComboBox(self)
        self.fpsCombo.setObjectName("cameraSettingsFpsCombo")
        fps_values = default_camera_fps_values()
        if int(fps or 30) not in fps_values:
            fps_values.append(int(fps or 30))
            fps_values.sort()
        for value in fps_values:
            self.fpsCombo.addItem(f"{value} fps", value)
        fps_index = self.fpsCombo.findData(int(fps or 30))
        self.fpsCombo.setCurrentIndex(max(0, fps_index))
        form.addRow("Frame rate", self.fpsCombo)

        self.resolutionCombo = QComboBox(self)
        self.resolutionCombo.setObjectName("cameraSettingsResolutionCombo")
        for label, res_width, res_height in default_camera_resolutions():
            self.resolutionCombo.addItem(label, (res_width, res_height))
        resolution_index = self.resolutionCombo.findData((width, height))
        self.resolutionCombo.setCurrentIndex(max(0, resolution_index))
        form.addRow("Resolution", self.resolutionCombo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, object]:
        width, height = self.resolutionCombo.currentData()
        return {
            "device": int(self.cameraCombo.currentData()),
            "frames": int(self.framesSpin.value()),
            "fps": int(self.fpsCombo.currentData()),
            "width": width,
            "height": height,
        }
