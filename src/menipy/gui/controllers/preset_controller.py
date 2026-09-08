"""Capture, validate and explicitly apply reusable analysis presets."""

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QMessageBox, QPushButton

from menipy.gui.services.sop_service import Sop
from menipy.models.preset import AnalysisPreset

CONTROLS = (
    "needleLengthSpin",
    "dropDensitySpin",
    "fluidDensitySpin",
    "gravitySpin",
    "substrateAngleSpin",
    "pendantNeedleIdSpin",
    "sessileBaselineModeCombo",
    "oscillatingFrequencySpin",
    "oscillatingAmplitudeSpin",
    "dynamicFpsSpin",
    "capillaryTubeDiameterSpin",
    "capillaryContactAngleSpin",
)


def plugin_fingerprints(window):
    """Identify the configured plugin source environment without importing code."""
    result = {}
    for directory in window.settings.plugin_dirs:
        for path in sorted(Path(directory).glob("*.py")):
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class PresetController:
    def __init__(self, window):
        self.window = window
        self.setup = window.setup_panel_ctrl
        self.sop = self.setup.sop_ctrl
        buttons = QHBoxLayout()
        self.setup.sopGroup.layout().addLayout(buttons)
        for title, callback in (
            ("Apply…", self.apply_selected),
            ("Update", self.update),
            ("Import…", self.import_file),
            ("Export…", self.export_file),
        ):
            button = QPushButton(title)
            button.setAccessibleName(f"{title.rstrip('…')} analysis preset")
            button.clicked.connect(callback)
            buttons.addWidget(button)

    def capture(self, name):
        parameters = self.window.pipeline_ctrl._build_pipeline_run_kwargs()[2]
        controls = {}
        for name_ in CONTROLS:
            widget = getattr(self.setup, name_, None)
            if widget is not None:
                controls[name_] = (
                    widget.value() if hasattr(widget, "value") else widget.currentText()
                )
        return AnalysisPreset(
            name=name,
            pipeline=self.setup.current_pipeline_name(),
            application_version=version("menipy"),
            stages=self.setup.collect_included_stages(),
            unit_system=self.window.settings.unit_system,
            preprocessing=self.window.preprocessing_ctrl.settings.model_copy(deep=True),
            edge_detection=self.window.edge_detection_ctrl.settings.model_copy(
                deep=True
            ),
            markers=self.window.preprocessing_ctrl.markers.model_copy(deep=True),
            controls=controls,
            pipeline_settings=dict(
                getattr(self.window.settings, "pipeline_settings", {}) or {}
            ),
            geometry={
                key: parameters[key]
                for key in ("roi", "needle_rect", "contact_line")
                if parameters.get(key) is not None
            },
            plugins=plugin_fingerprints(self.window),
        )

    def selected(self):
        sop = self.window.sops.get(
            self.setup.current_pipeline_name(), self.sop._selected_sop_key()
        )
        if sop is None or "__preset__" not in (sop.params or {}):
            raise ValueError(
                "This is a legacy stage-only SOP. Use Add or Update to save a complete preset."
            )
        return AnalysisPreset.model_validate(sop.params["__preset__"])

    def save(self, preset):
        self.window.sops.upsert(
            preset.pipeline,
            Sop(
                preset.name,
                preset.stages,
                {"__preset__": preset.model_dump(mode="json")},
            ),
        )
        self.sop._refresh_sop_combo(select=preset.name)

    def update(self):
        try:
            name = self.sop._selected_sop_key()
            if name == self.window.sops.default_name():
                raise ValueError(
                    "Use Add to create a named preset; the pipeline default is unchanged."
                )
            self.save(self.capture(name))
        except Exception as exc:
            QMessageBox.warning(self.window, "Preset", str(exc))

    def validate(self, preset):
        if preset.pipeline not in self.window.pipeline_ctrl.pipeline_map:
            raise ValueError(f"Pipeline is unavailable: {preset.pipeline}")
        if preset.pipeline not in {
            name
            for button, name in self.setup._pipeline_button_map.items()
            if button is not None
        }:
            raise ValueError(
                f"Pipeline is not selectable in this window: {preset.pipeline}"
            )
        if not preset.stages or set(preset.stages) - set(self.sop.stage_order):
            raise ValueError("Preset contains unavailable pipeline stages.")
        for key, value in preset.geometry.items():
            if key not in ("roi", "needle_rect", "contact_line"):
                raise ValueError(f"Unsupported preset geometry: {key}")
            try:
                values = (
                    [v for point in value for v in point]
                    if key == "contact_line"
                    else value
                )
                if len(values) != 4 or not all(math.isfinite(float(v)) for v in values):
                    raise ValueError()
                if key != "contact_line" and (values[2] <= 0 or values[3] <= 0):
                    raise ValueError()
            except (ValueError, TypeError):
                raise ValueError(f"Invalid preset geometry: {key}") from None
        if set(preset.controls) != {
            name for name in CONTROLS if getattr(self.setup, name, None) is not None
        }:
            raise ValueError(
                "Preset setup controls are incomplete or incompatible with this version."
            )
        for name, value in preset.controls.items():
            widget = getattr(self.setup, name)
            if hasattr(widget, "value"):
                if (
                    not isinstance(value, (int, float))
                    or not widget.minimum() <= value <= widget.maximum()
                ):
                    raise ValueError(f"Unsupported value for {name}: {value}")
            elif widget.findText(str(value)) < 0:
                raise ValueError(f"Unavailable option for {name}: {value}")
        available = plugin_fingerprints(self.window)
        missing = sorted(set(preset.plugins) - set(available))
        changed = [
            name
            for name in preset.plugins
            if name in available and available[name] != preset.plugins[name]
        ]
        if missing or changed:
            raise ValueError(
                f"Plugin environment conflict. Missing: {', '.join(missing) or 'none'}. Changed versions: {', '.join(changed) or 'none'}."
            )

    def apply_selected(self):
        try:
            self.apply(self.selected(), confirm=True)
        except Exception as exc:
            QMessageBox.warning(self.window, "Preset", str(exc))

    def apply(self, preset, *, confirm=True):
        self.validate(preset)
        summary = (
            f"Apply '{preset.name}' to {preset.pipeline}?\n"
            f"Stages: {', '.join(preset.stages)}\n"
            f"Units: {preset.unit_system}; edge method: {preset.edge_detection.method}\n"
            f"Geometry: {json.dumps(preset.geometry)}\n"
            "Preprocessing, detection, physics, calibration choices, markers and pipeline settings will be replaced. "
            "Pixel geometry will apply to the current source; review its fit before running.\n"
            f"Saved with Menipy {preset.application_version}; installed: {version('menipy')}."
        )
        if (
            confirm
            and QMessageBox.question(
                self.window,
                "Apply analysis preset",
                summary,
                QMessageBox.Apply | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            != QMessageBox.Apply
        ):
            return False
        for button, name in self.setup._pipeline_button_map.items():
            if name == preset.pipeline and button is not None:
                button.click()
                break
        self.window.main_controller.change_unit_system(preset.unit_system)
        for name, value in preset.controls.items():
            widget = getattr(self.setup, name)
            if hasattr(widget, "setValue"):
                widget.setValue(value)
            else:
                widget.setCurrentText(str(value))
        self.window.settings.pipeline_settings = dict(preset.pipeline_settings)
        self.window.settings.save()
        self.window.preprocessing_ctrl.set_settings(
            preset.preprocessing.model_copy(deep=True)
        )
        self.window.edge_detection_ctrl.set_settings(
            preset.edge_detection.model_copy(deep=True)
        )
        self.window.preprocessing_ctrl.update_markers(
            preset.markers.model_copy(deep=True)
        )
        self.window.preview_panel.clear_overlays()
        self.window.preprocessing_ctrl.clear_geometry()
        self.window._last_calibration_result = None
        from menipy.common.auto_calibrator import CalibrationResult

        geometry = preset.geometry
        result = CalibrationResult(
            roi_rect=geometry.get("roi"),
            needle_rect=geometry.get("needle_rect"),
            substrate_line=geometry.get("contact_line"),
            manual_regions=["roi", "needle", "substrate"],
            confidence_scores={"roi": 1.0, "needle": 1.0, "substrate": 1.0},
        )
        self.window.main_controller._on_calibration_complete(result)
        for widget in self.sop._step_widgets:
            widget.set_included(widget.step_name in preset.stages)
        self.window.pipeline_ctrl.mark_setup_changed()
        return True

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Import analysis preset", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            preset = AnalysisPreset.model_validate_json(
                Path(path).read_text(encoding="utf-8")
            )
            self.validate(preset)
            if self.window.sops.get(preset.pipeline, preset.name):
                raise ValueError(
                    "A preset with this name already exists. Rename it in the import file to keep both."
                )
            self.save(preset)
            QMessageBox.information(
                self.window,
                "Preset imported",
                f"Saved '{preset.name}' under {preset.pipeline}. Select that mode and use Apply to review it.",
            )
        except Exception as exc:
            QMessageBox.warning(self.window, "Import failed", str(exc))

    def export_file(self):
        try:
            preset = self.selected()
            path, _ = QFileDialog.getSaveFileName(
                self.window,
                "Export analysis preset",
                preset.name + ".json",
                "JSON (*.json)",
            )
            if path:
                Path(path).write_text(
                    preset.model_dump_json(indent=2), encoding="utf-8"
                )
        except Exception as exc:
            QMessageBox.warning(self.window, "Export failed", str(exc))
