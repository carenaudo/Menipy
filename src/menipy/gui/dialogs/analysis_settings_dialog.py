"""
Analysis Settings Dialog

Generic dialog that wraps waiting for pipeline steps selection and configuration.
"""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any, Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from menipy.gui.dialogs.geometry_config_dialog import GeometryConfigDialog
from menipy.gui.dialogs.overlay_config_dialog import OverlayConfigDialog
from menipy.gui.dialogs.preprocessing_config_dialog import PreprocessingConfigDialog
from menipy.models.config import (
    EdgeDetectionSettings,
    PreprocessingSettings,
)
from menipy.pipelines.base import PipelineBase
from menipy.pipelines.discover import PIPELINE_MAP


class AnalysisSettingsDialog(QDialog):
    """Configuration tabs for the stages of one pipeline.

    Which stages run is chosen in the setup panel's step list; this dialog only
    configures them.
    """

    def __init__(
        self,
        pipeline_name: str,
        *,
        preprocessing: PreprocessingSettings | None = None,
        edge: EdgeDetectionSettings | None = None,
        pipeline_settings: dict | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Analysis Settings")
        self.resize(800, 600)  # Slightly larger for more tabs

        self._pipeline_name = pipeline_name
        self._settings_store = QSettings("Menipy", "ADSA")

        # Load persisted settings if available (merging with passed args)
        saved = self._load_saved()
        self._preproc = saved.get("preproc") or preprocessing or PreprocessingSettings()
        self._edge = saved.get("edge") or edge or EdgeDetectionSettings()
        # The caller's per-pipeline settings are what runs use (they may come
        # from an applied preset); QSettings only fills in when there are none.
        self._pipeline_settings_dict = pipeline_settings or saved.get("pipeline") or {}

        # Load other settings from dict if present, or defaults. Densities and
        # gravity have no tab here: they come from the setup panel only.
        self._geometry_config = self._pipeline_settings_dict.get("geometry_config", {})
        self._overlay_config = self._pipeline_settings_dict.get("overlay_config", {})

        # One configuration tab per configurable stage the pipeline runs.
        # Which stages run is chosen in the setup panel's step list, not here.
        self._pipeline_class = PIPELINE_MAP.get(self._pipeline_name.lower())
        stage_source = self._pipeline_class or PipelineBase
        self._available_stages = stage_source.stage_names()

        self._tabs_map = {}  # Map stage name -> QWidget tab

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, 1)
        self._create_stage_tabs()

        # Footer buttons
        buttons = QHBoxLayout()
        buttons.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setProperty("secondary", True)
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Save")
        btn_ok.clicked.connect(self.accept)
        buttons.addWidget(btn_cancel)
        buttons.addWidget(btn_ok)
        layout.addLayout(buttons)

    def _create_stage_tabs(self):
        """Iterate over available stages and create tabs for relevant ones."""
        # Define a consistent order or use the pipeline order
        # We can use _available_stages for order

        seen_stages = set()

        for stage in self._available_stages:
            if stage in seen_stages:
                continue  # Avoid dups

            seen_stages.add(stage)
            tab_widget = None
            tab_title = stage.replace("_", " ").title()

            if stage == "preprocessing":
                tab_widget = self._build_preproc_tab()

            elif stage in ("contour_extraction", "edge_detection"):
                # Share the same tab builder
                tab_widget = self._build_edge_tab()
                tab_title = "Edge Detection"  # Nicer name

            elif stage == "geometric_features":
                tab_widget = self._build_geometry_tab()
                tab_title = "Geometry"

            elif stage == "overlay":
                tab_widget = self._build_overlay_tab()
                tab_title = "Overlay"

            elif stage == "contour_refinement":
                # For now optional, maybe just a placeholder or simple settings
                # Leaving blank/None means no tab created for this stage
                pass

            elif stage == "calibration":
                # Calibration usually happens via wizard, but maybe manual params?
                pass

            # Check for generic/custom if not handled above
            if tab_widget is None:
                # Try optional custom builder or pipeline-specific overrides?
                # For now, skip unhandled steps
                pass

            if tab_widget:
                self._tabs.addTab(tab_widget, tab_title)
                self._tabs_map[stage] = tab_widget

        # Finally, Pipeline-specific custom tab (always added if exists)
        self._pipeline_widget = self._build_pipeline_custom_tab()
        if self._pipeline_widget:
            self._tabs.addTab(
                self._pipeline_widget, f"{self._pipeline_name.title()} specific"
            )
            # Map this to a special key
            self._tabs_map["__pipeline_custom__"] = self._pipeline_widget

    # --- Tab Builders ---

    def _build_preproc_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        desc = QLabel("Configure preprocessing options.")
        v.addWidget(desc)

        summary = QLabel(self._preproc_summary())
        summary.setObjectName("preprocSummary")
        summary.setStyleSheet("color: #7f8c8d;")
        v.addWidget(summary)

        btn = QPushButton("Configure Preprocessing...")
        btn.clicked.connect(lambda: self._open_preproc_dialog(summary))
        v.addWidget(btn)
        v.addStretch(1)
        return w

    def _build_edge_tab(self) -> QWidget:
        """Edge detection tab with Method selector and Configuration button."""
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)

        # Import specific registry for edge detectors
        from menipy.common.registry import EDGE_DETECTORS

        # Get available methods from registry, fallback to defaults if empty
        methods = sorted(EDGE_DETECTORS.keys())
        if not methods:
            methods = [
                "canny",
                "sobel",
                "scharr",
                "laplacian",
                "threshold",
                "active_contour",
            ]

        self._edge_method = QComboBox()
        self._edge_method.addItems(methods)
        self._edge_method.setCurrentText(self._edge.method)
        form.addRow("Method", self._edge_method)

        # Button to open full configuration dialog
        btn_config = QPushButton("Configure Parameters...")
        btn_config.clicked.connect(self._configure_edge_parameters)
        form.addRow("", btn_config)

        # Add some summary text or status if desired?
        # For now, just the button as requested.

        return w

    def _configure_edge_parameters(self):
        """Open a specific configuration dialog for the selected method."""
        # Ensure current method selection is synced
        method = self._edge_method.currentText()
        self._edge.method = method

        from menipy.common.plugin_settings import get_detector_settings_model
        from menipy.gui.dialogs.plugin_config_dialog import PluginConfigDialog

        plugin_model = get_detector_settings_model(method)

        if not plugin_model:
            # Fallback for unconfigured methods
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(
                self, "Configuration", f"No specific settings for '{method}'."
            )
            return

        # Prepare defaults from legacy fields to ensure UI reflects current state
        defaults = {}
        s = self._edge
        if method == "canny":
            defaults = {
                "threshold1": s.canny_threshold1,
                "threshold2": s.canny_threshold2,
                "aperture_size": s.canny_aperture_size,
                "L2gradient": s.canny_L2_gradient,
            }
        elif method == "threshold":
            defaults = {
                "threshold_value": s.threshold_value,
                "max_value": s.threshold_max_value,
                "type": s.threshold_type,
            }
        elif method == "sobel":
            defaults = {
                "kernel_size": s.sobel_kernel_size,
                "threshold_value": s.threshold_value,
                "max_value": s.threshold_max_value,
            }
        elif method == "scharr":
            defaults = {
                "threshold_value": s.threshold_value,
                "max_value": s.threshold_max_value,
            }
        elif method == "laplacian":
            defaults = {
                "kernel_size": s.laplacian_kernel_size,
                "threshold_value": s.threshold_value,
                "max_value": s.threshold_max_value,
            }
        elif method in ("legacy_snake", "active_contour", "improved_snake"):
            defaults = {
                "iterations": s.snake_iterations,
                "alpha": s.snake_alpha,
                "beta": s.snake_beta,
                "gamma": s.snake_gamma,
            }

        # Resolve current settings: plugin specific takes precedence?
        # Actually, legacy fields are the "true" state for core methods currently.
        # So defaults (legacy) should override stored plugin_settings if present,
        # unless we consider plugin_settings the new truth?
        # Let's assume legacy fields are strict master for now.

        saved_data = s.plugin_settings.get(method, {})
        # Start with saved, update with legacy defaults
        current_data = saved_data.copy()
        current_data.update(defaults)

        dlg = PluginConfigDialog(plugin_model, current_data, parent=self)
        if dlg.exec():
            new_settings = dlg.get_settings()

            # 1. Update plugin_settings dict
            if self._edge.plugin_settings is None:
                self._edge.plugin_settings = {}
            self._edge.plugin_settings[method] = new_settings

            # 2. Scync BACK to legacy fields
            if method == "canny":
                s.canny_threshold1 = new_settings.get("threshold1", s.canny_threshold1)
                s.canny_threshold2 = new_settings.get("threshold2", s.canny_threshold2)
                s.canny_aperture_size = new_settings.get(
                    "aperture_size", s.canny_aperture_size
                )
                s.canny_L2_gradient = new_settings.get(
                    "L2gradient", s.canny_L2_gradient
                )
            elif method == "threshold":
                s.threshold_value = new_settings.get(
                    "threshold_value", s.threshold_value
                )
                s.threshold_max_value = new_settings.get(
                    "max_value", s.threshold_max_value
                )
                s.threshold_type = new_settings.get("type", s.threshold_type)
            elif method == "sobel":
                s.sobel_kernel_size = new_settings.get(
                    "kernel_size", s.sobel_kernel_size
                )
                s.threshold_value = new_settings.get(
                    "threshold_value", s.threshold_value
                )
                s.threshold_max_value = new_settings.get(
                    "max_value", s.threshold_max_value
                )
            elif method == "scharr":
                s.threshold_value = new_settings.get(
                    "threshold_value", s.threshold_value
                )
                s.threshold_max_value = new_settings.get(
                    "max_value", s.threshold_max_value
                )
            elif method == "laplacian":
                s.laplacian_kernel_size = new_settings.get(
                    "kernel_size", s.laplacian_kernel_size
                )
                s.threshold_value = new_settings.get(
                    "threshold_value", s.threshold_value
                )
                s.threshold_max_value = new_settings.get(
                    "max_value", s.threshold_max_value
                )
            elif method in ("legacy_snake", "active_contour", "improved_snake"):
                s.snake_iterations = new_settings.get("iterations", s.snake_iterations)
                s.snake_alpha = new_settings.get("alpha", s.snake_alpha)
                s.snake_beta = new_settings.get("beta", s.snake_beta)
                s.snake_gamma = new_settings.get("gamma", s.snake_gamma)

    def _build_geometry_tab(self) -> QWidget:
        """Tab for Geometry settings using GeometryConfigDialog."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)

        lbl = QLabel("Configure geometry and detector options.")
        v.addWidget(lbl)

        self._geometry_summary = QLabel(f"Config: {len(self._geometry_config)} keys")
        self._geometry_summary.setStyleSheet("color: #7f8c8d;")
        v.addWidget(self._geometry_summary)

        btn = QPushButton("Configure Geometry...")
        btn.clicked.connect(self._open_geometry_dialog)
        v.addWidget(btn)
        v.addStretch(1)
        return w

    def _build_overlay_tab(self) -> QWidget:
        """Tab for Overlay settings using OverlayConfigDialog."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)

        lbl = QLabel("Configure result overlay styling.")
        v.addWidget(lbl)

        self._overlay_summary = QLabel("Overlay style configured.")
        self._overlay_summary.setStyleSheet("color: #7f8c8d;")
        v.addWidget(self._overlay_summary)

        btn = QPushButton("Configure Overlay...")
        btn.clicked.connect(self._open_overlay_dialog)
        v.addWidget(btn)
        v.addStretch(1)
        return w

    def _build_pipeline_custom_tab(self) -> QWidget | None:
        """Load pipeline-specific settings widget if available."""
        module_name = (
            f"menipy.gui.dialogs.analysis_settings.{self._pipeline_name}_settings"
        )
        try:
            mod = import_module(module_name)
            widget_cls = getattr(mod, "PipelineSettingsWidget", None)
            if widget_cls:
                self._pipeline_widget = widget_cls(
                    parent=self, settings=self._pipeline_settings_dict
                )
                return self._pipeline_widget
        except ModuleNotFoundError:
            pass
        except Exception:
            pass
        return None

    # --- Config Dialog Handlers ---

    def _open_preproc_dialog(self, summary_label: QLabel):
        dlg = PreprocessingConfigDialog(self._preproc, self)
        if dlg.exec():
            self._preproc = dlg.settings()
            summary_label.setText(self._preproc_summary())

    def _open_geometry_dialog(self):
        dlg = GeometryConfigDialog(parent=self)
        dlg.set_config(self._geometry_config)
        dlg.configApplied.connect(lambda cfg: setattr(self, "_geometry_config", cfg))
        if dlg.exec():
            # Already handled by signal or just update on accept
            pass
        # After exec, assume configApplied handled it or get it
        self._geometry_config = dlg.get_config()
        self._geometry_summary.setText(
            f"Detector: {self._geometry_config.get('detector')}"
        )

    def _open_overlay_dialog(self):
        """_open_overlay_dialog."""
        dlg = OverlayConfigDialog(parent=self)
        dlg.set_config(self._overlay_config)
        dlg.configApplied.connect(lambda cfg: setattr(self, "_overlay_config", cfg))
        if dlg.exec():
            pass
        self._overlay_config = dlg.get_config()

    def _preproc_summary(self) -> str:
        """_preproc_summary."""
        ad = "on" if self._preproc.auto_detect.enabled else "off"
        filt = self._preproc.filtering.method
        return f"Auto-detect: {ad} • Filter: {filt} • Resize: {self._preproc.resize.target_width or 'auto'}"

    # ------------------------------------------------------------------ persistence
    def _load_saved(self) -> dict:
        """_load_saved."""
        key = f"analysis/{self._pipeline_name}"
        raw = self._settings_store.value(key)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            pre = (
                PreprocessingSettings(**data.get("preproc", {}))
                if data.get("preproc")
                else None
            )
            edge = (
                EdgeDetectionSettings(**data.get("edge", {}))
                if data.get("edge")
                else None
            )
            pipe = data.get("pipeline") or {}
            # We don't unpack physics/overlay from 'pipeline' key here directly, caller init takes care of defaults
            return {"preproc": pre, "edge": edge, "pipeline": pipe}
        except Exception:
            return {}

    def persist(self):
        """Persist current selections to QSettings."""
        # Update pipeline settings with all sub-configs
        pipe_settings = self.pipeline_settings() or {}

        payload = {
            "preproc": self._preproc.model_dump(),
            "edge": self.edge_settings().model_dump(),
            "pipeline": pipe_settings,
        }
        key = f"analysis/{self._pipeline_name}"
        self._settings_store.setValue(key, json.dumps(payload))

    # ------------------------------------------------------------------ results
    def preprocessing_settings(self) -> PreprocessingSettings:
        """Preprocessing_settings."""
        return self._preproc

    def edge_settings(self) -> EdgeDetectionSettings:
        """Edge_settings."""
        # settings are updated via configuration dialog or internal state
        self._edge.method = self._edge_method.currentText()
        return self._edge

    def pipeline_settings(self) -> dict | None:
        """Settings of the pipeline-specific tab plus the dialog's sub-configs.

        Returns
        -------
        dict
            Flat pipeline settings for this pipeline.
        """
        settings = {}
        if self._pipeline_widget and hasattr(self._pipeline_widget, "get_settings"):
            settings = self._pipeline_widget.get_settings() or {}

        # Notes moved to presets; keep an old tab note until a new preset
        # takes it over (see PresetController.legacy_notes).
        legacy_notes = self._pipeline_settings_dict.get("notes")
        if legacy_notes:
            settings["notes"] = legacy_notes

        # Merge all our dynamic tabs
        settings["geometry_config"] = self._geometry_config
        settings["overlay_config"] = self._overlay_config

        return settings
