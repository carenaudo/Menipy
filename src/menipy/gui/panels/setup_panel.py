"""Controller for the setup panel widgets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
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
from menipy.gui.controllers.pipeline_ui_manager import PipelineUIManager

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
    draw_mode_requested = Signal(int)
    clear_overlays_requested = Signal()
    run_all_requested = Signal()
    play_stage_requested = Signal(str)
    config_stage_requested = Signal(str)
    pipeline_changed = Signal(str)
    source_mode_changed = Signal(str)

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
        self._pipeline_keys = {str(k).lower() for k in (pipeline_keys or PIPELINE_MAP.keys()) if k}

        # Pipeline selection buttons (replacing combo boxes)
        self.sessileBtn: Optional[QPushButton] = panel.findChild(QPushButton, "sessileBtn")
        self.pendantBtn: Optional[QPushButton] = panel.findChild(QPushButton, "pendantBtn")
        self.oscillatingBtn: Optional[QPushButton] = panel.findChild(QPushButton, "oscillatingBtn")
        self.capillaryBtn: Optional[QPushButton] = panel.findChild(QPushButton, "capillaryBtn")
        self.captiveBtn: Optional[QPushButton] = panel.findChild(QPushButton, "captiveBtn")

        # Legacy combo boxes (kept for backward compatibility)
        self.testCombo: Optional[QComboBox] = panel.findChild(QComboBox, "testCombo")
        self.pipelineCombo: Optional[QComboBox] = panel.findChild(QComboBox, "pipelineCombo")
        self.sopCombo: Optional[QComboBox] = panel.findChild(QComboBox, "sopCombo")
        self.addSopBtn: Optional[QToolButton] = panel.findChild(QToolButton, "addSopBtn")

        self.singleModeRadio: Optional[QRadioButton] = panel.findChild(QRadioButton, "singleModeRadio")
        self.batchModeRadio: Optional[QRadioButton] = panel.findChild(QRadioButton, "batchModeRadio")
        self.cameraModeRadio: Optional[QRadioButton] = panel.findChild(QRadioButton, "cameraModeRadio")

        self.imagePathEdit: Optional[QLineEdit] = panel.findChild(QLineEdit, "imagePathEdit")
        self.batchPathEdit: Optional[QLineEdit] = panel.findChild(QLineEdit, "batchPathEdit")
        self.sourceIdCombo: Optional[QComboBox] = panel.findChild(QComboBox, "sourceIdCombo")

        self.browseBtn: Optional[QToolButton] = panel.findChild(QToolButton, "browseBtn")
        self.batchBrowseBtn: Optional[QToolButton] = panel.findChild(QToolButton, "batchBrowseBtn")
        self.previewBtn: Optional[QToolButton] = panel.findChild(QToolButton, "previewBtn")

        # Calibration input widgets
        self.needleLengthSpin: Optional[QDoubleSpinBox] = panel.findChild(QDoubleSpinBox, "needleLengthSpin")
        self.dropDensitySpin: Optional[QDoubleSpinBox] = panel.findChild(QDoubleSpinBox, "dropDensitySpin")
        self.fluidDensitySpin: Optional[QDoubleSpinBox] = panel.findChild(QDoubleSpinBox, "fluidDensitySpin")
        self.substrateAngleSpin: Optional[QDoubleSpinBox] = panel.findChild(QDoubleSpinBox, "substrateAngleSpin")

        self.framesSpin: Optional[QSpinBox] = panel.findChild(QSpinBox, "framesSpin")

        self.stepsList: Optional[QListWidget] = panel.findChild(QListWidget, "stepsList")
        self.runAllBtn: Optional[QPushButton] = panel.findChild(QPushButton, "runAllBtn")

        # Pipeline UI Manager for dynamic configuration
        self.pipeline_ui_manager = PipelineUIManager()

        self.sop_ctrl = SopController(
            window=self.window,
            sops=self.sops,
            stage_order=self.stage_order,
            step_item_cls=self.step_item_cls,
            steps_list=self.stepsList,
            sop_combo=self.sopCombo,
            pipeline_getter=self.current_pipeline_name,
            pipeline_changed_callback=lambda pipeline: self.pipeline_changed.emit(pipeline),
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

        # Create pipeline button group for mutual exclusivity
        self._pipeline_group = QButtonGroup(self)
        self._pipeline_map: dict[QPushButton, str] = {}
        for btn, pipeline in (
            (self.sessileBtn, "sessile"),
            (self.pendantBtn, "pendant"),
            (self.oscillatingBtn, "oscillating"),
            (self.capillaryBtn, "capillary_rise"),
            (self.captiveBtn, "captive_bubble"),
        ):
            if btn:
                self._pipeline_group.addButton(btn)
                self._pipeline_map[btn] = pipeline

        if self.singleModeRadio and not self.singleModeRadio.isChecked():
            self.singleModeRadio.setChecked(True)

        if self.sourceIdCombo:
            self.sourceIdCombo.setEditable(False)

        self._populate_pipeline_combo()
        self._restore_settings()
        self.sop_ctrl.initialize()
        self._apply_theme_aware_button_styling()
        self._wire_controls()
        self._apply_mode(self._mode, emit=False)
        self._initial_pipeline_refresh()

    # -------------------------- public API --------------------------

    def current_pipeline_name(self) -> Optional[str]:
        """Gets the canonical pipeline name from button selection or combo box fallback."""
        # Check pipeline buttons first
        button_map = {
            self.sessileBtn: "sessile",
            self.pendantBtn: "pendant",
            self.oscillatingBtn: "oscillating",
            self.capillaryBtn: "capillary_rise",
            self.captiveBtn: "captive_bubble"
        }

        for button, pipeline_name in button_map.items():
            if button and button.isChecked():
                return pipeline_name

        # Fallback to legacy combo boxes
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
        }

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
        if hasattr(self, "sop_ctrl"):
            return self.sop_ctrl.collect_included_stages()
        return list(self.stage_order)

    def get_calibration_params(self) -> dict[str, float]:
        """Get current calibration parameter values for the selected pipeline."""
        pipeline_name = self.current_pipeline_name()
        if not pipeline_name:
            # Fallback to basic parameters if no pipeline selected
            return {
                "needle_length_mm": float(self.needleLengthSpin.value()) if self.needleLengthSpin else 10.0,
                "drop_density_kg_m3": float(self.dropDensitySpin.value()) if self.dropDensitySpin else 1000.0,
                "fluid_density_kg_m3": float(self.fluidDensitySpin.value()) if self.fluidDensitySpin else 1.2,
            }

        # Get pipeline-specific calibration parameters
        required_params = self.pipeline_ui_manager.get_calibration_params(pipeline_name)

        # Map UI widgets to parameter values based on pipeline requirements
        params = {}
        for param_name in required_params:
            if param_name == "needle_length_mm" and self.needleLengthSpin:
                params[param_name] = float(self.needleLengthSpin.value())
            elif param_name == "needle_diameter_mm" and self.needleLengthSpin:
                # Reuse needle length spin for diameter (could add separate widget later)
                params[param_name] = float(self.needleLengthSpin.value())
            elif param_name == "drop_density_kg_m3" and self.dropDensitySpin:
                params[param_name] = float(self.dropDensitySpin.value())
            elif param_name == "fluid_density_kg_m3" and self.fluidDensitySpin:
                params[param_name] = float(self.fluidDensitySpin.value())
            elif param_name == "tube_diameter_mm" and self.needleLengthSpin:
                # Reuse needle length spin for tube diameter
                params[param_name] = float(self.needleLengthSpin.value())
            elif param_name == "contact_angle_deg":
                # Default contact angle if no widget yet
                params[param_name] = 0.0
            elif param_name == "substrate_contact_angle_deg" and self.substrateAngleSpin:
                params[param_name] = float(self.substrateAngleSpin.value())
            else:
                # Default values for unknown parameters
                if "density" in param_name:
                    params[param_name] = 1000.0 if "drop" in param_name else 1.2
                elif "length" in param_name or "diameter" in param_name:
                    params[param_name] = 10.0
                elif "angle" in param_name:
                    params[param_name] = 0.0
                else:
                    params[param_name] = 0.0

        return params

    # -------------------------- internal helpers --------------------------

    def _apply_theme_aware_button_styling(self) -> None:
        """Apply theme-aware styling to pipeline selection buttons."""
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QPalette
        
        # Get the application palette to detect light/dark theme
        palette = QApplication.palette()
        is_dark_theme = palette.color(QPalette.ColorRole.Window).lightness() < 128
        
        # Define base background and text colors based on theme
        if is_dark_theme:
            # Use a dark background for dark themes
            base_bg = "#2b2b2b"  # Dark gray background
            base_text = "#e0e0e0"  # Light text
        else:
            base_bg = "#f8f9fa"  # Light gray for light themes
            base_text = "#212529"  # Dark text for light backgrounds
        
        # Button color schemes (accent color when checked)
        button_styles = {
            self.sessileBtn: "#4A90E2",  # Blue
            self.pendantBtn: "#7ED321",  # Green
            self.oscillatingBtn: "#F5A623",  # Orange
            self.capillaryBtn: "#9B59B6",  # Purple
            self.captiveBtn: "#50E3C2",  # Cyan
        }
        
        for button, accent_color in button_styles.items():
            if button:
                style = f"""
                    QPushButton {{
                        text-align: center;
                        padding: 8px;
                        border: 2px solid {accent_color};
                        border-radius: 6px;
                        background-color: {base_bg};
                        color: {base_text};
                    }}
                    QPushButton:checked {{
                        background-color: {accent_color};
                        color: white;
                        font-weight: bold;
                    }}
                    QPushButton:hover:!checked {{
                        border: 3px solid {accent_color};
                    }}
                """
                button.setStyleSheet(style)


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
        if self.imagePathEdit and getattr(self.settings, "last_image_path", None):
            self.imagePathEdit.setText(self.settings.last_image_path)
        selected = getattr(self.settings, "selected_pipeline", None)
        combo = self.testCombo or self.pipelineCombo
        if combo and selected:
            index = combo.findData(selected)
            if index != -1:
                combo.setCurrentIndex(index)

    def _wire_controls(self) -> None:
        if self.browseBtn:
            self.browseBtn.clicked.connect(self.browse_requested.emit)
        if self.batchBrowseBtn:
            self.batchBrowseBtn.clicked.connect(self.browse_batch_requested.emit)
        if self.previewBtn:
            self.previewBtn.clicked.connect(self.preview_requested.emit)
        # Note: Drawing tools removed in simplified calibration interface
        # The "Run All" button will now trigger the simple analysis.
        if self.runAllBtn:
            self.runAllBtn.setText("Analyze")
            self.runAllBtn.clicked.connect(self.analyze_requested.emit)
        if self.addSopBtn:
            self.addSopBtn.clicked.connect(self.sop_ctrl.on_add_sop)
        # Connect pipeline buttons to selection handler
        button_map = {
            self.sessileBtn: "sessile",
            self.pendantBtn: "pendant",
            self.oscillatingBtn: "oscillating",
            self.capillaryBtn: "capillary_rise",
            self.captiveBtn: "captive_bubble"
        }

        for button, pipeline_name in button_map.items():
            if button:
                button.clicked.connect(lambda checked, name=pipeline_name: self._on_pipeline_button_clicked(name))

        # Legacy combo box connections (for backward compatibility)
        combo = self.testCombo or self.pipelineCombo
        if combo:
            combo.currentTextChanged.connect(self.sop_ctrl.on_pipeline_changed)
        if self.imagePathEdit:
            self.imagePathEdit.textChanged.connect(lambda _: self._refresh_source_items())
        if self.batchPathEdit:
            self.batchPathEdit.textChanged.connect(lambda _: self._refresh_source_items())
        if self.sourceIdCombo:
            self.sourceIdCombo.currentTextChanged.connect(self._on_combo_text_changed)
        self._mode_group.buttonToggled.connect(self._on_mode_toggled)

    def _on_pipeline_button_clicked(self, pipeline_name: str) -> None:
        """Handle pipeline button clicks."""
        # Update SOP controller with new pipeline
        self.sop_ctrl.on_pipeline_changed(pipeline_name)

        # Update calibration UI based on pipeline requirements
        self._update_calibration_ui_for_pipeline(pipeline_name)

        # Emit pipeline changed signal
        self.pipeline_changed.emit(pipeline_name)

    def _update_calibration_ui_for_pipeline(self, pipeline_name: str) -> None:
        """Update calibration UI elements based on pipeline requirements."""
        if not self.panel:
            return

        # Get pipeline-specific calibration parameters
        required_params = self.pipeline_ui_manager.get_calibration_params(pipeline_name)

        # Update labels and visibility based on required parameters
        needle_label = self.panel.findChild(QWidget, "needleLengthLabel")
        needle_spin = self.needleLengthSpin
        drop_label = self.panel.findChild(QWidget, "dropDensityLabel")
        drop_spin = self.dropDensitySpin
        fluid_label = self.panel.findChild(QWidget, "fluidDensityLabel")
        fluid_spin = self.fluidDensitySpin
        substrate_label = self.panel.findChild(QWidget, "substrateAngleLabel")
        substrate_spin = self.substrateAngleSpin

        # Show/hide and relabel widgets based on pipeline needs
        if "needle_length_mm" in required_params:
            if needle_label:
                needle_label.setText("Needle Length (mm)")
                needle_label.show()
            if needle_spin:
                needle_spin.show()
        elif "needle_diameter_mm" in required_params:
            if needle_label:
                needle_label.setText("Needle Diameter (mm)")
                needle_label.show()
            if needle_spin:
                needle_spin.show()
        elif "tube_diameter_mm" in required_params:
            if needle_label:
                needle_label.setText("Tube Diameter (mm)")
                needle_label.show()
            if needle_spin:
                needle_spin.show()
        else:
            # Hide needle parameter if not needed
            if needle_label:
                needle_label.hide()
            if needle_spin:
                needle_spin.hide()

        # Drop density is common to most pipelines
        show_drop = "drop_density_kg_m3" in required_params
        if drop_label:
            drop_label.setVisible(show_drop)
        if drop_spin:
            drop_spin.setVisible(show_drop)

        # Fluid density is also common
        show_fluid = "fluid_density_kg_m3" in required_params
        if fluid_label:
            fluid_label.setVisible(show_fluid)
        if fluid_spin:
            fluid_spin.setVisible(show_fluid)

        # Substrate contact angle is specific to sessile drops
        show_substrate = "substrate_contact_angle_deg" in required_params
        if substrate_label:
            substrate_label.setVisible(show_substrate)
        if substrate_spin:
            substrate_spin.setVisible(show_substrate)

        # Update default values based on pipeline
        if needle_spin and needle_spin.isVisible():
            if "tube_diameter_mm" in required_params:
                needle_spin.setValue(5.0)  # Smaller default for tubes
            else:
                needle_spin.setValue(10.0)  # Standard needle size

        # Update spin box ranges based on parameter type
        if needle_spin and needle_spin.isVisible():
            if "tube_diameter_mm" in required_params:
                needle_spin.setRange(0.1, 50.0)  # Tube diameters
            elif "needle_diameter_mm" in required_params:
                needle_spin.setRange(0.1, 5.0)   # Needle diameters
            else:
                needle_spin.setRange(0.1, 100.0) # Needle lengths

    def _initial_pipeline_refresh(self) -> None:
        # Set default pipeline button (sessile) if no selection
        if self.sessileBtn and not any(btn.isChecked() for btn in [self.sessileBtn, self.pendantBtn, self.oscillatingBtn, self.capillaryBtn, self.captiveBtn] if btn):
            self.sessileBtn.setChecked(True)
            self._on_pipeline_button_clicked("sessile")

        # Legacy combo initialization
        combo = self.testCombo or self.pipelineCombo
        if combo:
            self.sop_ctrl.on_pipeline_changed(combo.currentText())

    def _on_mode_toggled(self, button: Optional[QRadioButton], checked: bool) -> None:
        if not checked or not button:
            return
        mode = self._mode_map.get(button)
        if mode:
            self._apply_mode(mode)

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
        single = self._mode == self.MODE_SINGLE
        batch = self._mode == self.MODE_BATCH
        camera = self._mode == self.MODE_CAMERA
        if self.imagePathEdit:
            self.imagePathEdit.setEnabled(single)
        if self.browseBtn:
            self.browseBtn.setEnabled(single)
        if self.batchPathEdit:
            self.batchPathEdit.setEnabled(batch)
        if self.batchBrowseBtn:
            self.batchBrowseBtn.setEnabled(batch)
        if self.framesSpin:
            self.framesSpin.setEnabled(camera)
        if self.sourceIdCombo:
            self.sourceIdCombo.setEditable(camera)
            self.sourceIdCombo.setEnabled(camera or single or batch)

    def _refresh_source_items(self) -> None:
        if self._mode == self.MODE_SINGLE:
            self._populate_single_selection()
        elif self._mode == self.MODE_BATCH:
            self._populate_batch_selection()
        else:
            self._populate_camera_selection()

    def _populate_single_selection(self) -> None:
        path = self.image_path()
        items = []
        if path:
            label = Path(path).name or path
            items.append((label, path))
        self._set_combo_items(items)

    def _populate_batch_selection(self) -> None:
        folder = self.batch_path()
        items = []
        if folder and Path(folder).is_dir():
            for child in sorted(Path(folder).iterdir()):
                if child.suffix.lower() in self._IMAGE_EXTENSIONS and child.is_file():
                    items.append((child.name, str(child)))
        self._set_combo_items(items)

    def _populate_camera_selection(self) -> None:
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
