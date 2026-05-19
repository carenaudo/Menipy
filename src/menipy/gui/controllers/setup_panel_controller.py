"""Controller for the setup panel widgets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from PySide6.QtCore import QObject, Signal, QTimer, Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QToolButton,
    QMainWindow,
    QWidget,
)

from menipy.gui.views.image_view import DRAW_POINT, DRAW_LINE, DRAW_RECT
from menipy.gui.controllers.sop_controller import SopController

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
    auto_calibrate_requested = Signal()

    def __init__(
        self,
        window: QMainWindow,
        panel: QWidget,
        settings: Any,
        sops: Any,
        stage_order: Sequence[str],
        step_item_cls: Optional[type],
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

        self.testCombo: Optional[QComboBox] = panel.findChild(QComboBox, "testCombo")
        self.pipelineCombo: Optional[QComboBox] = panel.findChild(
            QComboBox, "pipelineCombo"
        )
        self.sopCombo: Optional[QComboBox] = panel.findChild(QComboBox, "sopCombo")
        self.addSopBtn: Optional[QToolButton] = panel.findChild(
            QToolButton, "addSopBtn"
        )
        # Segmented pipeline selection buttons introduced in the revamped UI
        self.sessileBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "sessileBtn"
        )
        self.pendantBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "pendantBtn"
        )
        self.oscillatingBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "oscillatingBtn"
        )
        self.capillaryBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "capillaryBtn"
        )
        self.captiveBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "captiveBtn"
        )
        self._pipeline_button_map: dict[Optional[QPushButton], str] = {
            self.sessileBtn: "sessile",
            self.pendantBtn: "pendant",
            self.oscillatingBtn: "oscillating",
            self.capillaryBtn: "capillary_rise",
            self.captiveBtn: "captive_bubble",
        }

        self.singleModeRadio: Optional[QRadioButton] = panel.findChild(
            QRadioButton, "singleModeRadio"
        )
        self.batchModeRadio: Optional[QRadioButton] = panel.findChild(
            QRadioButton, "batchModeRadio"
        )
        self.cameraModeRadio: Optional[QRadioButton] = panel.findChild(
            QRadioButton, "cameraModeRadio"
        )

        self.imagePathEdit: Optional[QLineEdit] = panel.findChild(
            QLineEdit, "imagePathEdit"
        )
        self.batchPathEdit: Optional[QLineEdit] = panel.findChild(
            QLineEdit, "batchPathEdit"
        )
        self.sourceIdCombo: Optional[QComboBox] = panel.findChild(
            QComboBox, "sourceIdCombo"
        )
        self.labelImage: Optional[QLabel] = panel.findChild(QLabel, "labelImage")
        self.labelBatch: Optional[QLabel] = panel.findChild(QLabel, "labelBatch")
        self.labelSourceId: Optional[QLabel] = panel.findChild(QLabel, "labelSourceId")
        self.labelFrames: Optional[QLabel] = panel.findChild(QLabel, "labelFrames")

        self.browseBtn: Optional[QToolButton] = panel.findChild(
            QToolButton, "browseBtn"
        )
        self.batchBrowseBtn: Optional[QToolButton] = panel.findChild(
            QToolButton, "batchBrowseBtn"
        )
        self.previewBtn: Optional[QToolButton] = panel.findChild(
            QToolButton, "previewBtn"
        )
        self.autoCalibrateBtn: Optional[QPushButton] = panel.findChild(QPushButton, "autoCalibrateBtn")

        self.drawPointBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "drawPointBtn"
        )
        self.drawLineBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "drawLineBtn"
        )
        self.drawRectBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "drawRectBtn"
        )
        self.clearOverlayBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "clearOverlayBtn"
        )

        self.framesSpin: Optional[QSpinBox] = panel.findChild(QSpinBox, "framesSpin")

        # Calibration labels for unit dynamic updates
        self.needleLengthLabel: Optional[QLabel] = panel.findChild(QLabel, "needleLengthLabel")
        self.dropDensityLabel: Optional[QLabel] = panel.findChild(QLabel, "dropDensityLabel")
        self.fluidDensityLabel: Optional[QLabel] = panel.findChild(QLabel, "fluidDensityLabel")
        self.substrateAngleLabel: Optional[QLabel] = panel.findChild(QLabel, "substrateAngleLabel")
        
        self.needleLengthSpin: Optional[QDoubleSpinBox] = panel.findChild(QDoubleSpinBox, "needleLengthSpin")
        self.dropDensitySpin: Optional[QDoubleSpinBox] = panel.findChild(QDoubleSpinBox, "dropDensitySpin")
        self.fluidDensitySpin: Optional[QDoubleSpinBox] = panel.findChild(QDoubleSpinBox, "fluidDensitySpin")
        self.substrateAngleSpin: Optional[QDoubleSpinBox] = panel.findChild(QDoubleSpinBox, "substrateAngleSpin")
        self.sopGroup: Optional[QGroupBox] = panel.findChild(QGroupBox, "sopGroup")
        self.stepsGroup: Optional[QGroupBox] = panel.findChild(QGroupBox, "stepsGroup")
        self.advancedToggleBtn: Optional[QToolButton] = panel.findChild(
            QToolButton, "advancedToggleBtn"
        )
        if self.advancedToggleBtn is None:
            self.advancedToggleBtn = QToolButton(panel)
            self.advancedToggleBtn.setObjectName("advancedToggleBtn")
            self.advancedToggleBtn.setText("Advanced")
            self.advancedToggleBtn.setCheckable(True)
            self.advancedToggleBtn.setToolButtonStyle(Qt.ToolButtonTextOnly)
            root_layout = panel.layout()
            if root_layout is not None:
                insert_at = root_layout.indexOf(self.sopGroup) if self.sopGroup else -1
                if insert_at < 0:
                    insert_at = max(0, root_layout.count() - 1)
                root_layout.insertWidget(insert_at, self.advancedToggleBtn)

        steps_list_widget: Optional[QListWidget] = panel.findChild(
            QListWidget, "stepsList"
        )
        self.stepsList = steps_list_widget
        self.runAllBtn: Optional[QPushButton] = panel.findChild(
            QPushButton, "runAllBtn"
        )

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

        if self.singleModeRadio and not self.singleModeRadio.isChecked():
            self.singleModeRadio.setChecked(True)

        if self.sourceIdCombo:
            self.sourceIdCombo.setEditable(False)

        self._populate_pipeline_combo()
        self._restore_settings()
        self.sop_ctrl.initialize()
        self._wire_controls()
        self.set_advanced_visible(
            bool(getattr(self.settings, "advanced_ui_visible", False)), save=False
        )
        self._apply_mode(self._mode, emit=False)
        self._initial_pipeline_refresh()

    # -------------------------- public API --------------------------

    def current_pipeline_name(self) -> Optional[str]:
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

    def gather_run_params(self) -> dict[str, Any]:
        mode = self._mode
        frames = int(self.framesSpin.value()) if self.framesSpin else 1
        selected = self._selected_source_value()

        image: Optional[str] = None
        batch_folder: Optional[str] = None
        cam_id: Optional[int] = None

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
            
        return {
            "needle_diameter_mm": needle_diameter,
            "drop_density_kg_m3": drop_rho,
            "fluid_density_kg_m3": fluid_rho,
        }

    def refresh_ui_labels(self) -> None:
        """Updates setup panel labels based on the current unit system."""
        system = getattr(self.settings, "unit_system", "SI")
        
        if system == "SI":
            if self.needleLengthLabel: self.needleLengthLabel.setText("Needle Diameter (mm)")
            if self.dropDensityLabel: self.dropDensityLabel.setText("Drop Density (kg/m³)")
            if self.fluidDensityLabel: self.fluidDensityLabel.setText("Fluid Density (kg/m³)")
        else:
            if self.needleLengthLabel: self.needleLengthLabel.setText("Needle Diameter (cm)")
            if self.dropDensityLabel: self.dropDensityLabel.setText("Drop Density (g/cm³)")
            if self.fluidDensityLabel: self.fluidDensityLabel.setText("Fluid Density (g/cm³)")
            
        # Optional: convert current spinbox values to stay consistent when units toggle
        # (This is tricky if done multiple times without precision loss, but helpful for UX)
        # For now, just refreshing the labels as requested.

    def image_path(self) -> Optional[str]:
        if self.imagePathEdit and hasattr(self.imagePathEdit, "text"):
            text = self.imagePathEdit.text().strip()
            return text or None
        return None

    def batch_path(self) -> Optional[str]:
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
        """Show or hide advanced SOP/stage controls."""
        if self.advancedToggleBtn:
            self.advancedToggleBtn.blockSignals(True)
            self.advancedToggleBtn.setChecked(visible)
            self.advancedToggleBtn.setText("Advanced -" if visible else "Advanced +")
            self.advancedToggleBtn.blockSignals(False)
        for widget in (self.sopGroup, self.stepsGroup):
            if widget:
                widget.setVisible(visible)
        if save:
            self.settings.advanced_ui_visible = visible
            try:
                self.settings.save()
            except OSError:
                pass

        # -------------------------- internal helpers --------------------------

    def _populate_pipeline_combo(self) -> None:
        combo = self.testCombo or self.pipelineCombo
        if not combo:
            return
        combo.blockSignals(True)
        combo.clear()
        for key in sorted(list(self._pipeline_keys)):
            display_name = key.replace("_", " ").title()
            combo.addItem(display_name, userData=key)
        combo.blockSignals(False)

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
            self.autoCalibrateBtn.clicked.connect(lambda: self.auto_calibrate_requested.emit())
        if self.advancedToggleBtn:
            self.advancedToggleBtn.toggled.connect(
                lambda checked: self.set_advanced_visible(bool(checked))
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
            self.addSopBtn.clicked.connect(
                lambda: getattr(self.sop_ctrl, "on_add_sop")()
            )
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
                btn.clicked.connect(lambda _checked=False, m=mode: self._apply_mode(m, emit=True))
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

    def _select_pipeline_button(self, pipeline_name: Optional[str]) -> bool:
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
        return found

    def _on_pipeline_button_clicked(self, pipeline_name: str) -> None:
        """Handle clicks from the segmented pipeline buttons."""
        self._sync_combo_to_pipeline(pipeline_name)
        self.sop_ctrl.on_pipeline_changed(pipeline_name)
        self._update_pipeline_specific_visibility(pipeline_name)

    def _on_pipeline_combo_changed(self, text: str) -> None:
        """Keep segmented buttons in sync when legacy combo changes."""
        pipeline = (text or "").strip().lower().replace(" ", "_")
        if pipeline and pipeline not in self._pipeline_keys:
            pipeline = None
        self._select_pipeline_button(pipeline)
        self.sop_ctrl.on_pipeline_changed(text)
        self._update_pipeline_specific_visibility(pipeline)

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
            return

        combo = self.testCombo or self.pipelineCombo
        if combo:
            self.sop_ctrl.on_pipeline_changed(combo.currentText())
            self._update_pipeline_specific_visibility(self.current_pipeline_name())

    def _on_mode_toggled(self, button: Optional[QRadioButton], checked: bool) -> None:
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
        if mode == self._mode and not emit:
            pass
        elif mode == self._mode and emit:
            self.source_mode_changed.emit(mode)
        else:
            self._mode = mode
            self.source_mode_changed.emit(mode)
        self._update_widget_states()
        self._refresh_source_items()

    def _update_widget_states(self) -> None:
        """_update_widget_states."""
        single = self._mode == self.MODE_SINGLE
        batch = self._mode == self.MODE_BATCH
        camera = self._mode == self.MODE_CAMERA
        if self.imagePathEdit:
            self.imagePathEdit.setEnabled(single)
            self.imagePathEdit.setVisible(single)
        if self.labelImage:
            self.labelImage.setVisible(single)
        if self.browseBtn:
            self.browseBtn.setEnabled(single)
            self.browseBtn.setVisible(single)
        if self.batchPathEdit:
            self.batchPathEdit.setEnabled(batch)
            self.batchPathEdit.setVisible(batch)
        if self.labelBatch:
            self.labelBatch.setVisible(batch)
        if self.batchBrowseBtn:
            # Keep batch browse enabled so tests and automation can access
            # this action regardless of the currently selected mode.
            try:
                self.batchBrowseBtn.setEnabled(True)
                self.batchBrowseBtn.setVisible(batch)
            except Exception:
                # Best effort - ignore widget errors when running headless tests
                pass
        if self.framesSpin:
            self.framesSpin.setEnabled(camera)
            self.framesSpin.setVisible(camera)
        if self.labelFrames:
            self.labelFrames.setVisible(camera)
        if self.sourceIdCombo:
            self.sourceIdCombo.setEditable(camera)
            self.sourceIdCombo.setEnabled(camera or batch)
            self.sourceIdCombo.setVisible(camera or batch)
        if self.labelSourceId:
            self.labelSourceId.setVisible(camera or batch)

    def _update_pipeline_specific_visibility(
        self, pipeline_name: Optional[str] = None
    ) -> None:
        is_sessile = (pipeline_name or self.current_pipeline_name()) == "sessile"
        for widget in (self.substrateAngleLabel, self.substrateAngleSpin):
            if widget:
                widget.setVisible(is_sessile)

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
        combo.blockSignals(True)
        combo.clear()
        combo.setEditable(True)
        combo.setEnabled(True)
        combo.setCurrentText(self._last_camera_id)
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

    def _selected_source_value(self) -> Optional[str]:
        """_selected_source_value."""
        if not self.sourceIdCombo:
            return None
        data = self.sourceIdCombo.currentData()
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
            self._last_camera_id = text.strip() or "0"
