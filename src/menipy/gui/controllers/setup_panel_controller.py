"""Controller for the setup panel widgets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QWidget,
)

from menipy.gui.controllers.sop_controller import SopController
from menipy.gui.helpers.icon_loader import set_button_icon
from menipy.gui.services.camera_service import (
    CameraConfig,
    CameraDevice,
    discover_available_cameras,
)
from menipy.gui.services.material_catalog_service import MaterialCatalogService
from menipy.gui.views.image_view import DRAW_LINE, DRAW_POINT, DRAW_RECT

try:
    from menipy.pipelines.discover import PIPELINE_MAP
except Exception:
    PIPELINE_MAP = {}


class SetupPanelController(QObject):
    """Owns the setup panel widgets and exposes high-level signals/actions."""

    MODE_SINGLE = "single"
    MODE_BATCH = "batch"
    MODE_CAMERA = "camera"

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    browse_requested = Signal()
    browse_batch_requested = Signal()
    preview_requested = Signal()
    analyze_requested = Signal()
    # Draw mode payloads can be any Python object; use object to avoid coercion issues.
    draw_mode_requested = Signal(object)
    clear_overlays_requested = Signal()
    run_all_requested = Signal()
    play_stage_requested = Signal(str)
    config_stage_requested = Signal(str)
    pipeline_changed = Signal(str)
    source_mode_changed = Signal(str)
    camera_config_changed = Signal()
    auto_calibrate_requested = Signal()
    advanced_requested = Signal()

    def __init__(
        self,
        window: QMainWindow,
        panel: QWidget,
        settings: Any,
        sops: Any,
        stage_order: Sequence[str],
        step_item_cls: type | None,
        pipeline_keys: Sequence[str],
    ) -> None:
        super().__init__(window)
        self.window = window
        self.panel = panel
        self.settings = settings
        self.sops = sops
        self.stage_order = list(stage_order)
        self.step_item_cls = step_item_cls
        self._pipeline_keys = {
            str(k).lower() for k in (pipeline_keys or PIPELINE_MAP.keys()) if k
        }

        self.pipelineGroup: QGroupBox | None = panel.findChild(
            QGroupBox, "pipelineGroup"
        )
        self.sourceGroup: QGroupBox | None = panel.findChild(QGroupBox, "sourceGroup")

        self.testCombo: QComboBox | None = panel.findChild(QComboBox, "testCombo")
        self.pipelineCombo: QComboBox | None = panel.findChild(
            QComboBox, "pipelineCombo"
        )
        self.sopCombo: QComboBox | None = panel.findChild(QComboBox, "sopCombo")
        self.addSopBtn: QToolButton | None = panel.findChild(QToolButton, "addSopBtn")
        # Segmented pipeline selection buttons introduced in the revamped UI
        self.sessileBtn: QPushButton | None = panel.findChild(QPushButton, "sessileBtn")
        self.pendantBtn: QPushButton | None = panel.findChild(QPushButton, "pendantBtn")
        self.oscillatingBtn: QPushButton | None = panel.findChild(
            QPushButton, "oscillatingBtn"
        )
        self.capillaryBtn: QPushButton | None = panel.findChild(
            QPushButton, "capillaryBtn"
        )
        self.captiveBtn: QPushButton | None = panel.findChild(QPushButton, "captiveBtn")
        self.dynamicSessileBtn: QPushButton | None = panel.findChild(
            QPushButton, "dynamicSessileBtn"
        )
        self._pipeline_button_map: dict[QPushButton | None, str] = {
            self.sessileBtn: "sessile",
            self.pendantBtn: "pendant",
            self.oscillatingBtn: "oscillating",
            self.capillaryBtn: "capillary_rise",
            self.captiveBtn: "captive_bubble",
            self.dynamicSessileBtn: "sessile_dynamic",
        }
        self._pipeline_labels = {
            "sessile": "Sessile",
            "pendant": "Pendant",
            "oscillating": "Osc.",
            "capillary_rise": "Capillary",
            "captive_bubble": "Captive",
            "sessile_dynamic": "Dynamic",
        }
        self._pipeline_icon_names = {
            "sessile": "sessile",
            "pendant": "pendant",
            "oscillating": "oscillating",
            "capillary_rise": "capillary_rise",
            "captive_bubble": "captive_bubble",
            "sessile_dynamic": "sessile",
        }

        self.singleModeRadio: QRadioButton | None = panel.findChild(
            QRadioButton, "singleModeRadio"
        )
        self.batchModeRadio: QRadioButton | None = panel.findChild(
            QRadioButton, "batchModeRadio"
        )
        self.cameraModeRadio: QRadioButton | None = panel.findChild(
            QRadioButton, "cameraModeRadio"
        )

        self.imagePathEdit: QLineEdit | None = panel.findChild(
            QLineEdit, "imagePathEdit"
        )
        self.batchPathEdit: QLineEdit | None = panel.findChild(
            QLineEdit, "batchPathEdit"
        )
        self.sourceIdCombo: QComboBox | None = panel.findChild(
            QComboBox, "sourceIdCombo"
        )
        self.labelImage: QLabel | None = panel.findChild(QLabel, "labelImage")
        self.labelBatch: QLabel | None = panel.findChild(QLabel, "labelBatch")
        self.labelSourceId: QLabel | None = panel.findChild(QLabel, "labelSourceId")
        self.labelFrames: QLabel | None = panel.findChild(QLabel, "labelFrames")

        self.browseBtn: QToolButton | None = panel.findChild(QToolButton, "browseBtn")
        self.batchBrowseBtn: QToolButton | None = panel.findChild(
            QToolButton, "batchBrowseBtn"
        )
        self.previewBtn: QToolButton | None = panel.findChild(QToolButton, "previewBtn")
        self.autoCalibrateBtn: QPushButton | None = panel.findChild(
            QPushButton, "autoCalibrateBtn"
        )

        self.drawPointBtn: QPushButton | None = panel.findChild(
            QPushButton, "drawPointBtn"
        )
        self.drawLineBtn: QPushButton | None = panel.findChild(
            QPushButton, "drawLineBtn"
        )
        self.drawRectBtn: QPushButton | None = panel.findChild(
            QPushButton, "drawRectBtn"
        )
        self.clearOverlayBtn: QPushButton | None = panel.findChild(
            QPushButton, "clearOverlayBtn"
        )

        self.framesSpin: QSpinBox | None = panel.findChild(QSpinBox, "framesSpin")

        # Calibration labels for unit dynamic updates
        self.needleLengthLabel: QLabel | None = panel.findChild(
            QLabel, "needleLengthLabel"
        )
        self.dropDensityLabel: QLabel | None = panel.findChild(
            QLabel, "dropDensityLabel"
        )
        self.fluidDensityLabel: QLabel | None = panel.findChild(
            QLabel, "fluidDensityLabel"
        )
        self.substrateAngleLabel: QLabel | None = panel.findChild(
            QLabel, "substrateAngleLabel"
        )
        self.gravityLabel: QLabel | None = panel.findChild(QLabel, "gravityLabel")

        self.needleLengthSpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "needleLengthSpin"
        )
        self.dropDensitySpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "dropDensitySpin"
        )
        self.fluidDensitySpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "fluidDensitySpin"
        )
        self.substrateAngleSpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "substrateAngleSpin"
        )
        self.gravitySpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "gravitySpin"
        )
        self.needleDbBtn: QToolButton | None = panel.findChild(
            QToolButton, "needleDbBtn"
        )
        self.dropDensityDbBtn: QToolButton | None = panel.findChild(
            QToolButton, "dropDensityDbBtn"
        )
        self.fluidDensityDbBtn: QToolButton | None = panel.findChild(
            QToolButton, "fluidDensityDbBtn"
        )
        self.pipelineSettingsGroup: QGroupBox | None = panel.findChild(
            QGroupBox, "pipelineSettingsGroup"
        )
        self.pendantNeedleIdLabel: QLabel | None = panel.findChild(
            QLabel, "pendantNeedleIdLabel"
        )
        self.pendantNeedleIdSpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "pendantNeedleIdSpin"
        )
        self.sessileBaselineModeLabel: QLabel | None = panel.findChild(
            QLabel, "sessileBaselineModeLabel"
        )
        self.sessileBaselineModeCombo: QComboBox | None = panel.findChild(
            QComboBox, "sessileBaselineModeCombo"
        )
        self.oscillatingFrequencyLabel: QLabel | None = panel.findChild(
            QLabel, "oscillatingFrequencyLabel"
        )
        self.oscillatingFrequencySpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "oscillatingFrequencySpin"
        )
        self.oscillatingAmplitudeLabel: QLabel | None = panel.findChild(
            QLabel, "oscillatingAmplitudeLabel"
        )
        self.oscillatingAmplitudeSpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "oscillatingAmplitudeSpin"
        )
        self.dynamicFpsLabel: QLabel | None = panel.findChild(QLabel, "dynamicFpsLabel")
        self.dynamicFpsSpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "dynamicFpsSpin"
        )
        self.capillaryTubeDiameterLabel: QLabel | None = panel.findChild(
            QLabel, "capillaryTubeDiameterLabel"
        )
        self.capillaryTubeDiameterSpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "capillaryTubeDiameterSpin"
        )
        self.capillaryContactAngleLabel: QLabel | None = panel.findChild(
            QLabel, "capillaryContactAngleLabel"
        )
        self.capillaryContactAngleSpin: QDoubleSpinBox | None = panel.findChild(
            QDoubleSpinBox, "capillaryContactAngleSpin"
        )
        self.captiveDetectionLabel: QLabel | None = panel.findChild(
            QLabel, "captiveDetectionLabel"
        )
        self.captiveDetectionValue: QLabel | None = panel.findChild(
            QLabel, "captiveDetectionValue"
        )
        self.sopGroup: QGroupBox | None = panel.findChild(QGroupBox, "sopGroup")
        self.stepsGroup: QGroupBox | None = panel.findChild(QGroupBox, "stepsGroup")
        self.advancedToggleBtn: QToolButton | None = panel.findChild(
            QToolButton, "advancedToggleBtn"
        )
        if self.advancedToggleBtn is None:
            self.advancedToggleBtn = QToolButton(panel)
            self.advancedToggleBtn.setObjectName("advancedToggleBtn")
            self.advancedToggleBtn.setText("Advanced")
            self.advancedToggleBtn.setCheckable(False)
            self.advancedToggleBtn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            root_layout = panel.layout()
            if root_layout is not None:
                insert_at = root_layout.indexOf(self.sopGroup) if self.sopGroup else -1
                if insert_at < 0:
                    insert_at = max(0, root_layout.count() - 1)
                root_layout.insertWidget(insert_at, self.advancedToggleBtn)

        steps_list_widget: QListWidget | None = panel.findChild(
            QListWidget, "stepsList"
        )
        self.stepsList = steps_list_widget
        self.runAllBtn: QPushButton | None = panel.findChild(QPushButton, "runAllBtn")

        self.sop_ctrl = SopController(
            window=self.window,
            sops=self.sops,
            stage_order=self.stage_order,
            step_item_cls=self.step_item_cls,
            steps_list=steps_list_widget,
            sop_combo=self.sopCombo,
            pipeline_getter=self.current_pipeline_name,
            pipeline_changed_callback=lambda pipeline: self.pipeline_changed.emit(
                pipeline
            ),
            play_callback=self.play_stage_requested.emit,
            config_callback=self.config_stage_requested.emit,
        )

        self._mode_group = QButtonGroup(self)
        self._mode_map: dict[QRadioButton, str] = {}
        for btn, mode in (
            (self.singleModeRadio, self.MODE_SINGLE),
            (self.batchModeRadio, self.MODE_BATCH),
            (self.cameraModeRadio, self.MODE_CAMERA),
        ):
            if btn:
                self._mode_group.addButton(btn)
                self._mode_map[btn] = mode
        self._mode = self.MODE_SINGLE
        self._last_camera_id = "0"
        self._camera_devices: list[CameraDevice] = [CameraDevice(0, "Camera 0")]
        self._camera_frames = int(self.framesSpin.value()) if self.framesSpin else 1
        self._camera_fps = 30
        self._camera_width: int | None = None
        self._camera_height: int | None = None
        self._workflow_source_stack_mode = False
        self._catalog_service = MaterialCatalogService()

        if self.singleModeRadio and not self.singleModeRadio.isChecked():
            self.singleModeRadio.setChecked(True)

        if self.sourceIdCombo:
            self.sourceIdCombo.setEditable(False)
        for density_spin in (self.dropDensitySpin, self.fluidDensitySpin):
            if density_spin:
                density_spin.setDecimals(max(3, density_spin.decimals()))

        self._populate_pipeline_combo()
        self._restore_settings()
        self.sop_ctrl.initialize()
        self._apply_icons()
        self._wire_controls()
        self.set_advanced_visible(
            bool(getattr(self.settings, "advanced_ui_visible", False)), save=False
        )
        self._apply_mode(self._mode, emit=False)
        self._initial_pipeline_refresh()

    # -------------------------- public API --------------------------

    def current_pipeline_name(self) -> str | None:
        """Gets the canonical pipeline name from the UI selection."""
        # Prefer segmented buttons if present
        for button, pipeline in self._pipeline_button_map.items():
            if button and button.isChecked():
                return pipeline

        combo = self.testCombo or self.pipelineCombo
        if not combo:
            return None
        data = combo.currentData()
        if isinstance(data, str) and data.lower() in self._pipeline_keys:
            return data.lower()
        text = combo.currentText().lower()
        return text if text in self._pipeline_keys else None

    def current_mode(self) -> str:
        return self._mode

    def set_workflow_source_stack_mode(self, enabled: bool) -> None:
        """Let the workbench own source visibility through its stacked host."""
        self._workflow_source_stack_mode = bool(enabled)
        self._update_widget_states()

    def camera_devices(self) -> list[CameraDevice]:
        return list(self._camera_devices)

    def camera_config(self) -> CameraConfig:
        try:
            device = int(self._last_camera_id)
        except (TypeError, ValueError):
            device = 0
        return CameraConfig(
            device=device,
            fps=int(self._camera_fps or 30),
            width=self._camera_width,
            height=self._camera_height,
        )

    def camera_settings_values(self) -> dict[str, Any]:
        config = self.camera_config()
        return {
            "device": config.device,
            "frames": int(self._camera_frames or 1),
            "fps": config.fps,
            "width": config.width,
            "height": config.height,
        }

    def apply_camera_settings(
        self,
        *,
        device: int,
        frames: int,
        fps: int,
        width: int | None,
        height: int | None,
    ) -> None:
        self._last_camera_id = str(int(device))
        self._camera_frames = max(1, int(frames or 1))
        self._camera_fps = max(1, int(fps or 30))
        self._camera_width = int(width) if width else None
        self._camera_height = int(height) if height else None
        if self.framesSpin:
            self.framesSpin.setValue(self._camera_frames)
        self._populate_camera_selection()
        self.camera_config_changed.emit()

    def gather_run_params(self) -> dict[str, Any]:
        mode = self._mode
        frames = (
            int(
                self._camera_frames
                if mode == self.MODE_CAMERA
                else self.framesSpin.value()
            )
            if self.framesSpin
            else int(self._camera_frames or 1)
        )
        selected = self._selected_source_value()

        image: str | None = None
        batch_folder: str | None = None
        cam_id: int | None = None

        if mode == self.MODE_CAMERA:
            if selected is not None:
                try:
                    cam_id = int(str(selected).strip())
                    self._last_camera_id = str(cam_id)
                except ValueError:
                    cam_id = None
            image = None
        elif mode == self.MODE_BATCH:
            batch_folder = self.batch_path()
            image = selected
        else:
            image = self.image_path() or selected

        return {
            "name": self.current_pipeline_name(),
            "mode": mode,
            "use_camera": mode == self.MODE_CAMERA,
            "frames": frames,
            "image": image,
            "batch_folder": batch_folder,
            "cam_id": cam_id,
            "calibration_params": self.get_calibration_params(),
            "analysis_params": self.get_analysis_params(),
        }

    def get_calibration_params(self) -> dict[str, Any]:
        """Returns calibration parameters normalized to SI units."""
        from menipy.common.units import convert_to_si

        system = getattr(self.settings, "unit_system", "SI")

        # Use defaults if widgets missing
        needle_diameter = 0.54
        drop_rho = 1000.0
        fluid_rho = 1.2

        if self.needleLengthSpin:
            needle_val = self.needleLengthSpin.value()
            needle_diameter = convert_to_si(needle_val, "length", system)
        if self.dropDensitySpin:
            drop_val = self.dropDensitySpin.value()
            drop_rho = convert_to_si(drop_val, "density", system)
        if self.fluidDensitySpin:
            fluid_val = self.fluidDensitySpin.value()
            fluid_rho = convert_to_si(fluid_val, "density", system)
        gravity = self.gravitySpin.value() if self.gravitySpin else 9.80665

        return {
            "needle_diameter_mm": needle_diameter,
            "drop_density_kg_m3": drop_rho,
            "fluid_density_kg_m3": fluid_rho,
            "gravity_m_s2": gravity,
            "g": gravity,
        }

    def get_analysis_params(self) -> dict[str, Any]:
        """Return pipeline-specific setup values without changing public contracts."""
        pipeline = self.current_pipeline_name()
        params: dict[str, Any] = {"pipeline": pipeline}
        if self.substrateAngleSpin:
            params["substrate_contact_angle_deg"] = self.substrateAngleSpin.value()
        if self.pendantNeedleIdSpin:
            params["needle_inner_diameter_mm"] = self.pendantNeedleIdSpin.value()
        if self.sessileBaselineModeCombo:
            params["baseline_mode"] = (
                self.sessileBaselineModeCombo.currentText().lower()
            )
        if self.oscillatingFrequencySpin:
            params["oscillation_frequency_hz"] = self.oscillatingFrequencySpin.value()
        if self.oscillatingAmplitudeSpin:
            params["oscillation_amplitude_mm"] = self.oscillatingAmplitudeSpin.value()
        if self.dynamicFpsSpin:
            params["sequence_fps"] = self.dynamicFpsSpin.value()
        if self.capillaryTubeDiameterSpin:
            params["tube_diameter_mm"] = self.capillaryTubeDiameterSpin.value()
        if self.capillaryContactAngleSpin:
            params["capillary_contact_angle_deg"] = (
                self.capillaryContactAngleSpin.value()
            )
        return params

    def refresh_ui_labels(self) -> None:
        """Updates setup panel labels based on the current unit system."""
        system = getattr(self.settings, "unit_system", "SI")

        if system == "SI":
            if self.needleLengthLabel:
                self.needleLengthLabel.setText("Needle OD (mm)")
            if self.dropDensityLabel:
                self.dropDensityLabel.setText("Heavy phase rho")
            if self.fluidDensityLabel:
                self.fluidDensityLabel.setText("Light phase rho")
        else:
            if self.needleLengthLabel:
                self.needleLengthLabel.setText("Needle OD (cm)")
            if self.dropDensityLabel:
                self.dropDensityLabel.setText("Heavy phase rho")
            if self.fluidDensityLabel:
                self.fluidDensityLabel.setText("Light phase rho")

        # Optional: convert current spinbox values to stay consistent when units toggle
        # (This is tricky if done multiple times without precision loss, but helpful for UX)
        # For now, just refreshing the labels as requested.

    def image_path(self) -> str | None:
        if self.imagePathEdit and hasattr(self.imagePathEdit, "text"):
            text = self.imagePathEdit.text().strip()
            return text or None
        return None

    def batch_path(self) -> str | None:
        if self.batchPathEdit and hasattr(self.batchPathEdit, "text"):
            text = self.batchPathEdit.text().strip()
            return text or None
        return None

    def set_image_path(self, path: str) -> None:
        if self.imagePathEdit:
            self.imagePathEdit.setText(path)
            self._refresh_source_items()

    def set_batch_path(self, path: str) -> None:
        if self.batchPathEdit:
            self.batchPathEdit.setText(path)
            self._refresh_source_items()

    def set_camera_enabled(self, on: bool) -> None:
        if on:
            if self.cameraModeRadio:
                self.cameraModeRadio.setChecked(True)
            self._apply_mode(self.MODE_CAMERA)
        elif self._mode == self.MODE_CAMERA:
            if self.singleModeRadio:
                self.singleModeRadio.setChecked(True)
            self._apply_mode(self.MODE_SINGLE)

    def collect_included_stages(self) -> list[str]:
        """collect included stages.

        Returns
        -------
        type
        Description.
        """
        if hasattr(self, "sop_ctrl"):
            return self.sop_ctrl.collect_included_stages()
        return list(self.stage_order)

    def set_advanced_visible(self, visible: bool, *, save: bool = True) -> None:
        """Keep advanced SOP/stage controls out of the setup rail.

        The SOP widgets are hosted by AdvancedWorkflowDialog in the guided workbench.
        This compatibility method intentionally does not expand them inline anymore.
        """
        if self.advancedToggleBtn:
            self.advancedToggleBtn.blockSignals(True)
            self.advancedToggleBtn.setCheckable(False)
            self.advancedToggleBtn.setChecked(False)
            self.advancedToggleBtn.setText("Advanced")
            self.advancedToggleBtn.blockSignals(False)
        for widget in (self.sopGroup, self.stepsGroup):
            if widget:
                widget.setVisible(False)

        # -------------------------- internal helpers --------------------------

    def _populate_pipeline_combo(self) -> None:
        combo = self.testCombo or self.pipelineCombo
        if not combo:
            return
        combo.blockSignals(True)
        combo.clear()
        for key in sorted(self._pipeline_keys):
            display_name = key.replace("_", " ").title()
            combo.addItem(display_name, userData=key)
        combo.blockSignals(False)

    def _apply_icons(self) -> None:
        """Apply resource-backed icons to setup rail controls."""
        self._sync_pipeline_button_presentation()

        for button, icon_name in (
            (self.singleModeRadio, "file"),
            (self.batchModeRadio, "batch"),
            (self.cameraModeRadio, "camera"),
        ):
            set_button_icon(button, icon_name, size=15, clear_text=True)
            if button:
                button.setMinimumWidth(36)
                button.setMaximumWidth(44)
        if self.singleModeRadio:
            self.singleModeRadio.setToolTip("File")
        if self.batchModeRadio:
            self.batchModeRadio.setToolTip("Batch")
        if self.cameraModeRadio:
            self.cameraModeRadio.setToolTip("Camera")

        for button, icon_name in (
            (self.browseBtn, "file"),
            (self.batchBrowseBtn, "batch"),
            (self.previewBtn, "zoom-in"),
            (self.addSopBtn, "list"),
        ):
            set_button_icon(button, icon_name, size=15)
            if isinstance(button, QToolButton):
                button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        set_button_icon(self.autoCalibrateBtn, "info", size=15)
        if self.autoCalibrateBtn:
            self.autoCalibrateBtn.setText("Calibrate")
            self.autoCalibrateBtn.setMinimumHeight(34)
            self.autoCalibrateBtn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        if self.runAllBtn:
            self.runAllBtn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        set_button_icon(self.advancedToggleBtn, "settings", size=15)
        if isinstance(self.advancedToggleBtn, QToolButton):
            self.advancedToggleBtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            self.advancedToggleBtn.setCheckable(False)
            self.advancedToggleBtn.setMinimumHeight(34)
            self.advancedToggleBtn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        for button, tooltip in (
            (self.needleDbBtn, "Select needle from database"),
            (self.dropDensityDbBtn, "Select heavy phase material from database"),
            (self.fluidDensityDbBtn, "Select light phase material from database"),
        ):
            set_button_icon(button, "database", size=15, clear_text=True)
            if isinstance(button, QToolButton):
                button.setToolButtonStyle(Qt.ToolButtonIconOnly)
                button.setAutoRaise(True)
                button.setEnabled(True)
                button.setToolTip(tooltip)
                button.setFixedSize(30, 28)
                button.setStyleSheet("QToolButton { padding: 3px; }")

        set_button_icon(self.runAllBtn, "play", size=16)
        set_button_icon(self.drawPointBtn, "info", size=14)
        set_button_icon(self.drawLineBtn, "list", size=14)
        set_button_icon(self.drawRectBtn, "roi", size=14)
        set_button_icon(self.clearOverlayBtn, "x", size=14)

    def _sync_pipeline_button_presentation(self) -> None:
        """Show text only for the selected analysis button, icons otherwise."""
        for button, pipeline_name in self._pipeline_button_map.items():
            if button is None:
                continue
            label = self._pipeline_labels.get(pipeline_name, pipeline_name)
            button.setToolTip(label)
            if button.isChecked():
                button.setText(label)
                button.setIcon(QIcon())
                button.setMinimumWidth(76)
                button.setMaximumWidth(120)
            else:
                button.setText("")
                set_button_icon(
                    button,
                    self._pipeline_icon_names.get(pipeline_name, pipeline_name),
                    size=15,
                )
                button.setMinimumWidth(36)
                button.setMaximumWidth(42)

    def _restore_settings(self) -> None:
        """_restore_settings."""
        if self.imagePathEdit and getattr(self.settings, "last_image_path", None):
            self.imagePathEdit.setText(self.settings.last_image_path)
        selected = getattr(self.settings, "selected_pipeline", None)
        combo = self.testCombo or self.pipelineCombo
        if combo and selected:
            index = combo.findData(selected)
            if index != -1:
                combo.setCurrentIndex(index)
        if selected:
            self._select_pipeline_button(selected)

    def _wire_controls(self) -> None:
        """_wire_controls."""
        if self.browseBtn:
            self.browseBtn.clicked.connect(lambda: self.browse_requested.emit())
        if self.batchBrowseBtn:
            # Keep batch browse enabled and emit even if UI mode is single; tests
            # and some automation expect this button to be reachable.
            try:
                self.batchBrowseBtn.setEnabled(True)
            except Exception:
                pass
            self.batchBrowseBtn.clicked.connect(
                lambda: self.browse_batch_requested.emit()
            )
        if self.previewBtn:
            self.previewBtn.clicked.connect(lambda: self.preview_requested.emit())
        if self.autoCalibrateBtn:
            self.autoCalibrateBtn.clicked.connect(
                lambda: self.auto_calibrate_requested.emit()
            )
        if self.advancedToggleBtn:
            self.advancedToggleBtn.clicked.connect(
                lambda _checked=False: self.advanced_requested.emit()
            )
        if self.needleDbBtn:
            self.needleDbBtn.clicked.connect(self._select_needle_from_database)
        if self.dropDensityDbBtn:
            self.dropDensityDbBtn.clicked.connect(
                lambda _checked=False: self._select_material_from_database(
                    self.dropDensitySpin, "heavy phase"
                )
            )
        if self.fluidDensityDbBtn:
            self.fluidDensityDbBtn.clicked.connect(
                lambda _checked=False: self._select_material_from_database(
                    self.fluidDensitySpin, "light phase"
                )
            )
        if self.drawPointBtn:
            self.drawPointBtn.clicked.connect(
                lambda: self.draw_mode_requested.emit(DRAW_POINT)
            )
        if self.drawLineBtn:
            self.drawLineBtn.clicked.connect(
                lambda: self.draw_mode_requested.emit(DRAW_LINE)
            )
        if self.drawRectBtn:
            self.drawRectBtn.clicked.connect(
                lambda: self.draw_mode_requested.emit(DRAW_RECT)
            )
        if self.clearOverlayBtn:
            self.clearOverlayBtn.clicked.connect(self.clear_overlays_requested.emit)
        if self.runAllBtn:
            self.runAllBtn.clicked.connect(self.run_all_requested.emit)
        if self.addSopBtn:
            # Route through a lambda so tests can monkeypatch sop_ctrl.on_add_sop and observe the call
            self.addSopBtn.clicked.connect(lambda: self.sop_ctrl.on_add_sop())
        for button, pipeline_name in self._pipeline_button_map.items():
            if button:
                button.clicked.connect(
                    lambda _checked=False, name=pipeline_name: self._on_pipeline_button_clicked(
                        name
                    )
                )

        # Ensure radio button clicks emit mode change even if they are already selected
        for btn, mode in (
            (self.singleModeRadio, self.MODE_SINGLE),
            (self.batchModeRadio, self.MODE_BATCH),
            (self.cameraModeRadio, self.MODE_CAMERA),
        ):
            if btn:
                btn.clicked.connect(
                    lambda _checked=False, m=mode: self._apply_mode(m, emit=True)
                )
        combo = self.testCombo or self.pipelineCombo
        if combo:
            combo.currentTextChanged.connect(self._on_pipeline_combo_changed)

        # Debounce refresh calls triggered by text changes so tests that call setText
        # and then explicitly emit textChanged don't produce duplicate refreshes.
        def _schedule_refresh():
            """_schedule_refresh."""
            if getattr(self, "_refresh_scheduled", False):
                return
            self._refresh_scheduled = True
            QTimer.singleShot(20, self._run_scheduled_refresh)

        if self.imagePathEdit:
            self.imagePathEdit.textChanged.connect(lambda _: _schedule_refresh())
        if self.batchPathEdit:
            self.batchPathEdit.textChanged.connect(lambda _: _schedule_refresh())
        if self.sourceIdCombo:
            self.sourceIdCombo.currentTextChanged.connect(self._on_combo_text_changed)
        self._mode_group.buttonToggled.connect(self._on_mode_toggled)

    def _select_database_item(self, table_type: str) -> dict[str, Any] | None:
        """Open a database selector and return the accepted item."""
        from menipy.gui.dialogs.material_dialog import MaterialDialog

        selected: dict[str, Any] | None = None

        def _capture(item: dict) -> None:
            nonlocal selected
            selected = dict(item or {})

        dialog = MaterialDialog(self.window, selection_mode=True, table_type=table_type)
        dialog.item_selected.connect(_capture)
        if dialog.exec() == QDialog.Accepted:
            return selected
        return None

    def _on_needle_database_clicked(self) -> None:
        item = self._select_database_item("needles")
        if not item:
            return
        self._apply_database_value(
            item=item,
            field="outer_diameter",
            spinbox=self.needleLengthSpin,
            quantity="length",
            label="Needle OD",
        )

    def _on_drop_density_database_clicked(self) -> None:
        item = self._select_database_item("materials")
        if not item:
            return
        self._apply_database_value(
            item=item,
            field="density",
            spinbox=self.dropDensitySpin,
            quantity="density",
            label="Heavy phase density",
        )

    def _on_fluid_density_database_clicked(self) -> None:
        item = self._select_database_item("materials")
        if not item:
            return
        self._apply_database_value(
            item=item,
            field="density",
            spinbox=self.fluidDensitySpin,
            quantity="density",
            label="Light phase density",
        )

    def _apply_database_value(
        self,
        *,
        item: dict[str, Any],
        field: str,
        spinbox: QDoubleSpinBox | None,
        quantity: str,
        label: str,
    ) -> None:
        """Apply a selected database value to a setup spinbox."""
        if spinbox is None:
            return

        raw_value = item.get(field)
        if not isinstance(raw_value, (int, float)):
            QMessageBox.warning(
                self.window,
                "Database Selection",
                f"Selected item does not define {field.replace('_', ' ')}.",
            )
            return

        from menipy.common.units import convert_from_si

        system = getattr(self.settings, "unit_system", "SI")
        value = convert_from_si(float(raw_value), quantity, system)
        if value < spinbox.minimum():
            spinbox.setMinimum(value)
        if value > spinbox.maximum():
            spinbox.setMaximum(value)
        spinbox.setValue(value)

        status_bar = getattr(self.window, "statusBar", None)
        if callable(status_bar):
            try:
                name = str(item.get("name") or "database item")
                status_bar().showMessage(f"{label} set from {name}.", 2500)
            except Exception:
                pass

    def _show_status_message(self, message: str, timeout_ms: int = 2500) -> None:
        status_bar = getattr(self.window, "statusBar", None)
        if callable(status_bar):
            try:
                status_bar().showMessage(message, timeout_ms)
                return
            except Exception:
                pass

    def _select_needle_from_database(self) -> None:
        record = self._catalog_service.select_needle(self.window)
        if not record:
            self._show_status_message("No needle selected.", 1500)
            return
        self._apply_needle_record(record)

    def _select_material_from_database(
        self, target_spin: QDoubleSpinBox | None, phase_label: str
    ) -> None:
        record = self._catalog_service.select_material(self.window)
        if not record:
            self._show_status_message("No material selected.", 1500)
            return
        self._apply_material_record(record, target_spin, phase_label)

    def _apply_needle_record(self, record: dict[str, Any]) -> None:
        from menipy.common.units import convert_from_si

        system = getattr(self.settings, "unit_system", "SI")
        name = str(record.get("name") or record.get("gauge") or "needle")
        outer_diameter = self._number_from_record(record, "outer_diameter")
        inner_diameter = self._number_from_record(record, "inner_diameter")

        if outer_diameter is not None and self.needleLengthSpin:
            self.needleLengthSpin.setValue(
                convert_from_si(outer_diameter, "length", system)
            )
        if inner_diameter is not None and self.pendantNeedleIdSpin:
            self.pendantNeedleIdSpin.setValue(
                convert_from_si(inner_diameter, "length", system)
            )
        self._show_status_message(f"Selected needle: {name}", 2500)

    def _apply_material_record(
        self,
        record: dict[str, Any],
        target_spin: QDoubleSpinBox | None,
        phase_label: str,
    ) -> None:
        from menipy.common.units import convert_from_si

        density = self._number_from_record(record, "density")
        name = str(record.get("name") or "material")
        if density is None or target_spin is None:
            self._show_status_message(f"{name} has no density value.", 2500)
            return
        system = getattr(self.settings, "unit_system", "SI")
        target_spin.setValue(convert_from_si(density, "density", system))
        self._show_status_message(f"Selected {phase_label}: {name}", 2500)

    @staticmethod
    def _number_from_record(record: dict[str, Any], key: str) -> float | None:
        value = record.get(key)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _sync_combo_to_pipeline(self, pipeline_name: str) -> bool:
        """Ensure the legacy combo box reflects the segmented button selection."""
        combo = self.testCombo or self.pipelineCombo
        if not combo:
            return False
        index = combo.findData(pipeline_name)
        if index == -1:
            return False
        prev = combo.blockSignals(True)
        try:
            combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(prev)
        return True

    def _select_pipeline_button(self, pipeline_name: str | None) -> bool:
        """Select the matching segmented button without emitting extra signals."""
        if not pipeline_name:
            return False
        found = False
        for button, name in self._pipeline_button_map.items():
            if not button:
                continue
            if name == pipeline_name:
                if not button.isChecked():
                    button.setChecked(True)
                found = True
        self._sync_pipeline_button_presentation()
        return found

    def _on_pipeline_button_clicked(self, pipeline_name: str) -> None:
        """Handle clicks from the segmented pipeline buttons."""
        self._sync_combo_to_pipeline(pipeline_name)
        self.sop_ctrl.on_pipeline_changed(pipeline_name)
        self._update_pipeline_specific_visibility(pipeline_name)
        self._sync_pipeline_button_presentation()

    def _on_pipeline_combo_changed(self, text: str) -> None:
        """Keep segmented buttons in sync when legacy combo changes."""
        pipeline = (text or "").strip().lower().replace(" ", "_")
        if pipeline and pipeline not in self._pipeline_keys:
            pipeline = None
        self._select_pipeline_button(pipeline)
        self.sop_ctrl.on_pipeline_changed(text)
        self._update_pipeline_specific_visibility(pipeline)
        self._sync_pipeline_button_presentation()

    def _initial_pipeline_refresh(self) -> None:
        """_initial_pipeline_refresh."""
        pipeline = self.current_pipeline_name()
        if not pipeline:
            preferred = getattr(self.settings, "selected_pipeline", None)
            if not self._select_pipeline_button(preferred):
                self._select_pipeline_button("sessile")
            pipeline = self.current_pipeline_name()

        if pipeline:
            self.sop_ctrl.on_pipeline_changed(pipeline)
            self._update_pipeline_specific_visibility(pipeline)
            self._sync_pipeline_button_presentation()
            return

        combo = self.testCombo or self.pipelineCombo
        if combo:
            self.sop_ctrl.on_pipeline_changed(combo.currentText())
            self._update_pipeline_specific_visibility(self.current_pipeline_name())
            self._sync_pipeline_button_presentation()

    def _on_mode_toggled(self, button: QRadioButton | None, checked: bool) -> None:
        if not checked or not button:
            return
        mode = self._mode_map.get(button)
        if mode:
            self._apply_mode(mode)

    def _run_scheduled_refresh(self) -> None:
        """_run_scheduled_refresh."""
        try:
            self._refresh_source_items()
        finally:
            self._refresh_scheduled = False

    def _apply_mode(self, mode: str, emit: bool = True) -> None:
        if mode not in {self.MODE_SINGLE, self.MODE_BATCH, self.MODE_CAMERA}:
            mode = self.MODE_SINGLE
        changed = mode != self._mode
        if changed:
            self._mode = mode
        self._update_widget_states()
        self._refresh_source_items()
        if emit and (changed or mode == self._mode):
            self.source_mode_changed.emit(mode)

    def _update_widget_states(self) -> None:
        """_update_widget_states."""
        single = self._mode == self.MODE_SINGLE
        batch = self._mode == self.MODE_BATCH
        camera = self._mode == self.MODE_CAMERA
        stacked_source = bool(getattr(self, "_workflow_source_stack_mode", False))

        def set_mode_visible(widget: QWidget | None, visible: bool) -> None:
            if widget is not None and not stacked_source:
                widget.setVisible(visible)

        if self.imagePathEdit:
            self.imagePathEdit.setEnabled(single)
        set_mode_visible(self.imagePathEdit, single)
        set_mode_visible(self.labelImage, single)
        if self.browseBtn:
            self.browseBtn.setEnabled(single)
        set_mode_visible(self.browseBtn, single)
        if self.batchPathEdit:
            self.batchPathEdit.setEnabled(batch)
        set_mode_visible(self.batchPathEdit, batch)
        set_mode_visible(self.labelBatch, batch)
        if self.batchBrowseBtn:
            # Keep batch browse enabled so tests and automation can access
            # this action regardless of the currently selected mode.
            try:
                self.batchBrowseBtn.setEnabled(True)
            except Exception:
                # Best effort - ignore widget errors when running headless tests
                pass
        set_mode_visible(self.batchBrowseBtn, batch)
        if self.framesSpin:
            self.framesSpin.setEnabled(camera)
        set_mode_visible(self.framesSpin, camera)
        set_mode_visible(self.labelFrames, camera)
        if self.sourceIdCombo:
            self.sourceIdCombo.setEditable(False)
            self.sourceIdCombo.setEnabled(camera or batch)
        set_mode_visible(self.sourceIdCombo, camera or batch)
        set_mode_visible(self.labelSourceId, camera or batch)

    def _update_pipeline_specific_visibility(
        self, pipeline_name: str | None = None
    ) -> None:
        pipeline = pipeline_name or self.current_pipeline_name()
        is_sessile = pipeline in {"sessile", "sessile_dynamic"}
        for widget in (self.substrateAngleLabel, self.substrateAngleSpin):
            if widget:
                widget.setVisible(is_sessile)

        group_titles = {
            "pendant": "Pendant Settings",
            "sessile": "Sessile Settings",
            "oscillating": "Oscillating Settings",
            "capillary_rise": "Capillary Settings",
            "captive_bubble": "Captive Bubble Settings",
            "sessile_dynamic": "Dynamic Sessile Settings",
        }
        if self.pipelineSettingsGroup:
            self.pipelineSettingsGroup.setTitle(
                group_titles.get(str(pipeline), "Pipeline Settings")
            )

        visibility_sets = {
            "pendant": (self.pendantNeedleIdLabel, self.pendantNeedleIdSpin),
            "sessile": (
                self.sessileBaselineModeLabel,
                self.sessileBaselineModeCombo,
            ),
            "oscillating": (
                self.oscillatingFrequencyLabel,
                self.oscillatingFrequencySpin,
                self.oscillatingAmplitudeLabel,
                self.oscillatingAmplitudeSpin,
            ),
            "capillary_rise": (
                self.capillaryTubeDiameterLabel,
                self.capillaryTubeDiameterSpin,
                self.capillaryContactAngleLabel,
                self.capillaryContactAngleSpin,
            ),
            "captive_bubble": (
                self.captiveDetectionLabel,
                self.captiveDetectionValue,
            ),
            "sessile_dynamic": (self.dynamicFpsLabel, self.dynamicFpsSpin),
        }
        all_pipeline_widgets: list[QWidget] = []
        for widgets in visibility_sets.values():
            all_pipeline_widgets.extend(
                widget for widget in widgets if isinstance(widget, QWidget)
            )
        active_widgets = {
            widget
            for widget in visibility_sets.get(str(pipeline), ())
            if isinstance(widget, QWidget)
        }
        for widget in all_pipeline_widgets:
            widget.setVisible(widget in active_widgets)

    def _refresh_source_items(self) -> None:
        """_refresh_source_items."""
        if self._mode == self.MODE_SINGLE:
            self._populate_single_selection()
        elif self._mode == self.MODE_BATCH:
            self._populate_batch_selection()
        else:
            self._populate_camera_selection()

    def _populate_single_selection(self) -> None:
        """_populate_single_selection."""
        path = self.image_path()
        items = []
        if path:
            label = Path(path).name or path
            items.append((label, path))
        self._set_combo_items(items)

    def _populate_batch_selection(self) -> None:
        """_populate_batch_selection."""
        folder = self.batch_path()
        items = []
        if folder and Path(folder).is_dir():
            for child in sorted(Path(folder).iterdir()):
                if child.suffix.lower() in self._IMAGE_EXTENSIONS and child.is_file():
                    items.append((child.name, str(child)))
        self._set_combo_items(items)

    def _populate_camera_selection(self) -> None:
        """_populate_camera_selection."""
        if not self.sourceIdCombo:
            return
        combo = self.sourceIdCombo
        try:
            devices = discover_available_cameras()
        except Exception:
            devices = []
        self._camera_devices = devices or [CameraDevice(0, "Camera 0")]
        try:
            selected_device = int(self._last_camera_id or "0")
        except (TypeError, ValueError):
            selected_device = 0
        if all(camera.device != selected_device for camera in self._camera_devices):
            self._camera_devices.append(
                CameraDevice(selected_device, f"Camera {selected_device}")
            )
        combo.blockSignals(True)
        combo.clear()
        combo.setEditable(False)
        for camera in self._camera_devices:
            combo.addItem(camera.label, userData=int(camera.device))
        index = combo.findData(selected_device)
        combo.setCurrentIndex(max(0, index))
        data = combo.currentData()
        if isinstance(data, int):
            self._last_camera_id = str(data)
        combo.setEnabled(True)
        combo.blockSignals(False)

    def _set_combo_items(self, items: list[tuple[str, str]]) -> None:
        if not self.sourceIdCombo:
            return
        combo = self.sourceIdCombo
        combo.blockSignals(True)
        combo.clear()
        combo.setEditable(False)
        for label, value in items:
            combo.addItem(label, userData=value)
        combo.setEnabled(bool(items))
        if items:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _selected_source_value(self) -> str | None:
        """_selected_source_value."""
        if not self.sourceIdCombo:
            return None
        data = self.sourceIdCombo.currentData()
        if isinstance(data, int):
            return str(data)
        if isinstance(data, str) and data:
            return data
        text = self.sourceIdCombo.currentText().strip()
        if not text:
            return None
        if self._mode == self.MODE_BATCH:
            folder = self.batch_path()
            if folder:
                candidate = Path(folder) / text
                return str(candidate)
        if self._mode == self.MODE_SINGLE:
            image = self.image_path()
            if image:
                return image
        return text

    def _on_combo_text_changed(self, text: str) -> None:
        if self._mode == self.MODE_CAMERA:
            data = self.sourceIdCombo.currentData() if self.sourceIdCombo else None
            self._last_camera_id = str(
                data if isinstance(data, int) else (text.strip() or "0")
            )
            self.camera_config_changed.emit()
