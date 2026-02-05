"""
Analysis Settings Dialog

Generic dialog that wraps waiting for pipeline steps selection and configuration.
"""
from __future__ import annotations

from importlib import import_module
from typing import Optional, Any
import json

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QCheckBox,
    QScrollArea,
    QGroupBox,
)

from menipy.models.config import PreprocessingSettings, EdgeDetectionSettings, PhysicsParams
from menipy.gui.dialogs.preprocessing_config_dialog import PreprocessingConfigDialog
from menipy.gui.dialogs.physics_config_dialog import PhysicsConfigDialog
from menipy.gui.dialogs.geometry_config_dialog import GeometryConfigDialog
from menipy.gui.dialogs.overlay_config_dialog import OverlayConfigDialog
from menipy.pipelines.discover import PIPELINE_MAP


class AnalysisSettingsDialog(QDialog):
    """Collects analysis settings; now with Steps Choicer."""

    def __init__(
        self,
        pipeline_name: str,
        *,
        preprocessing: Optional[PreprocessingSettings] = None,
        edge: Optional[EdgeDetectionSettings] = None,
        pipeline_settings: Optional[dict] = None,
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
        self._pipeline_settings_dict = saved.get("pipeline") or pipeline_settings or {}
        
        # Load other settings from dict if present, or defaults
        self._physics_params = PhysicsParams(**self._pipeline_settings_dict.get("physics", {}))
        self._geometry_config = self._pipeline_settings_dict.get("geometry_config", {})
        self._overlay_config = self._pipeline_settings_dict.get("overlay_config", {})
        
        # Pipeline metadata for steps
        self._pipeline_class = PIPELINE_MAP.get(self._pipeline_name.lower())
        self._ui_metadata = getattr(self._pipeline_class, "ui_metadata", {}) if self._pipeline_class else {}
        
        # Determine available stages from pipeline metadata or fallback
        self._available_stages = self._ui_metadata.get("stages", [])
        if not self._available_stages and self._pipeline_class:
            # Fallback to DEFAULT_SEQ if no metadata
            self._available_stages = [n for n, _ in self._pipeline_class.DEFAULT_SEQ]
            
        # Determine enabled stages
        # Default to all if not specified in settings
        self._enabled_stages = set(
            self._pipeline_settings_dict.get("enabled_stages", self._available_stages)
        )

        self._tabs_map = {}  # Map stage name -> QWidget tab
        
        self._build_ui()
        self._update_tabs_visibility()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs, 1)

        # 1. Steps Choicer Tab (Always first)
        self._tabs.addTab(self._build_steps_tab(), "Steps")

        # Create tabs for all potential stages
        # We create them all once, then show/hide based on enabled status
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
                continue # Avoid dups
            
            seen_stages.add(stage)
            tab_widget = None
            tab_title = stage.replace("_", " ").title()
            
            if stage == "preprocessing":
                tab_widget = self._build_preproc_tab()
            
            elif stage in ("contour_extraction", "edge_detection"):
                 # Share the same tab builder
                 tab_widget = self._build_edge_tab()
                 tab_title = "Edge Detection" # Nicer name
                 
            elif stage == "geometric_features":
                tab_widget = self._build_geometry_tab()
                tab_title = "Geometry"
            
            elif stage == "physics":
                tab_widget = self._build_physics_tab()
                tab_title = "Physics"
                
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
            self._tabs.addTab(self._pipeline_widget, f"{self._pipeline_name.title()} specific")
            # Map this to a special key
            self._tabs_map["__pipeline_custom__"] = self._pipeline_widget


    def _build_steps_tab(self) -> QWidget:
        """Create the tab for selecting enabled pipeline steps."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        
        info = QLabel("Select the analysis steps to perform:")
        info.setStyleSheet("font-weight: bold; margin-bottom: 8px;")
        layout.addWidget(info)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        v = QVBoxLayout(content)
        v.setSpacing(10)
        
        stage_names = {
            "acquisition": "Image Acquisition",
            "preprocessing": "Preprocessing",
            "feature_detection": "Feature Detection",
            "contour_extraction": "Contour Extraction",
            "contour_refinement": "Contour Refinement",
            "calibration": "Calibration",
            "geometric_features": "Geometry",
            "physics": "Physics",
            "profile_fitting": "Profile Fitting",
            "compute_metrics": "Compute Metrics",
            "overlay": "Result Overlay",
            "validation": "Validation",
        }
        
        self._stage_checkboxes = {}
        
        for stage in self._available_stages:
            name = stage_names.get(stage, stage.replace("_", " ").title())
            chk = QCheckBox(name)
            chk.setChecked(stage in self._enabled_stages)
            chk.stateChanged.connect(lambda s, st=stage: self._on_step_toggled(st, s))
            
            if stage == "acquisition":
                chk.setEnabled(False)
                chk.setChecked(True)
                
            v.addWidget(chk)
            self._stage_checkboxes[stage] = chk
            
        v.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # Helper buttons
        btn_layout = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(self._select_all_steps)
        btn_none = QPushButton("Select Minimum")
        btn_none.clicked.connect(self._select_min_steps)
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_none)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return w

    def _on_step_toggled(self, stage: str, state: int):
        checked = (state == Qt.CheckState.Checked.value)
        if checked:
            self._enabled_stages.add(stage)
        else:
            self._enabled_stages.discard(stage)
        self._update_tabs_visibility()

    def _select_all_steps(self):
        for stage, chk in self._stage_checkboxes.items():
            if chk.isEnabled():
                chk.setChecked(True)

    def _select_min_steps(self):
        min_stages = {"acquisition"}
        for stage, chk in self._stage_checkboxes.items():
            if chk.isEnabled():
                chk.setChecked(stage in min_stages)

    def _update_tabs_visibility(self):
        """Show/Hide tabs based on enabled stages."""
        for stage, widget in self._tabs_map.items():
            if stage == "__pipeline_custom__":
                # Logic for custom tab? Maybe always show if enabled?
                # Or assume it relies on 'profile_fitting' or similar?
                # For now let's always show it if it exists
                continue
                
            idx = self._tabs.indexOf(widget)
            if idx >= 0:
                is_enabled = stage in self._enabled_stages
                # Edge detection alias
                if stage == "contour_extraction": 
                    is_enabled = is_enabled or ("edge_detection" in self._enabled_stages)
                
                self._tabs.setTabVisible(idx, is_enabled)
                
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
            methods = ["canny", "sobel", "scharr", "laplacian", "threshold", "active_contour"]
            
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
        current_method = self._edge_method.currentText()
        self._edge.method = current_method
        
        # Check if it's a plugin with a registered model
        from menipy.common.plugin_settings import get_detector_settings_model
        from menipy.gui.dialogs.plugin_config_dialog import PluginConfigDialog
        
        plugin_model = get_detector_settings_model(current_method)
        
        if plugin_model:
            # It's a plugin -> Use PluginConfigDialog
            # Get current plugin settings (or empty dict)
            current_data = self._edge.plugin_settings.get(current_method, {})
            
            dlg = PluginConfigDialog(plugin_model, current_data, parent=self)
            if dlg.exec():
                new_settings = dlg.get_settings()
                if self._edge.plugin_settings is None:
                    self._edge.plugin_settings = {}
                self._edge.plugin_settings[current_method] = new_settings
                
        else:
            # It's a core method (or plugin without model) -> Use EdgeDetectionConfigDialog in compact mode
            from menipy.gui.dialogs.edge_detection_config_dialog import EdgeDetectionConfigDialog
            # We assume EdgeDetectionConfigDialog now accepts 'compact_mode'
            try:
                dlg = EdgeDetectionConfigDialog(parent=self, settings=self._edge, compact_mode=True)
            except TypeError:
                 # Fallback if we haven't updated it yet (safety)
                dlg = EdgeDetectionConfigDialog(parent=self, settings=self._edge)
                
            if dlg.exec():
                self._edge = dlg.settings()
                # Sync back method
                if self._edge.method != current_method:
                    self._edge_method.setCurrentText(self._edge.method)
        
    def _build_physics_tab(self) -> QWidget:
        """Tab for PhysicsParams using PhysicsConfigDialog."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        
        lbl = QLabel("Configure physical parameters (density, gravity, etc).")
        v.addWidget(lbl)
        
        self._physics_summary = QLabel(str(self._physics_params)) # Just basic str for now
        self._physics_summary.setWordWrap(True)
        self._physics_summary.setStyleSheet("color: #7f8c8d;")
        v.addWidget(self._physics_summary)
        
        btn = QPushButton("Configure Physics...")
        btn.clicked.connect(self._open_physics_dialog)
        v.addWidget(btn)
        v.addStretch(1)
        return w
        
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
        
        self._overlay_summary = QLabel(f"Overlay style configured.")
        self._overlay_summary.setStyleSheet("color: #7f8c8d;")
        v.addWidget(self._overlay_summary)
        
        btn = QPushButton("Configure Overlay...")
        btn.clicked.connect(self._open_overlay_dialog)
        v.addWidget(btn)
        v.addStretch(1)
        return w

    def _build_pipeline_custom_tab(self) -> Optional[QWidget]:
        """Load pipeline-specific settings widget if available."""
        module_name = f"menipy.gui.dialogs.analysis_settings.{self._pipeline_name}_settings"
        try:
            mod = import_module(module_name)
            widget_cls = getattr(mod, "PipelineSettingsWidget", None)
            if widget_cls:
                self._pipeline_widget = widget_cls(parent=self, settings=self._pipeline_settings_dict)
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

    def _open_physics_dialog(self):
        dlg = PhysicsConfigDialog(self._physics_params, parent=self)
        if dlg.exec():
            self._physics_params = dlg.get_params()
            self._physics_summary.setText(str(self._physics_params))

    def _open_geometry_dialog(self):
        dlg = GeometryConfigDialog(parent=self)
        dlg.set_config(self._geometry_config)
        dlg.configApplied.connect(lambda cfg: setattr(self, '_geometry_config', cfg))
        if dlg.exec():
            # Already handled by signal or just update on accept
             pass
        # After exec, assume configApplied handled it or get it
        self._geometry_config = dlg.get_config()
        self._geometry_summary.setText(f"Detector: {self._geometry_config.get('detector')}")

    def _open_overlay_dialog(self):
        dlg = OverlayConfigDialog(parent=self)
        dlg.set_config(self._overlay_config)
        dlg.configApplied.connect(lambda cfg: setattr(self, '_overlay_config', cfg))
        if dlg.exec():
            pass
        self._overlay_config = dlg.get_config()


    def _preproc_summary(self) -> str:
        ad = "on" if self._preproc.auto_detect.enabled else "off"
        filt = self._preproc.filtering.method
        return f"Auto-detect: {ad} • Filter: {filt} • Resize: {self._preproc.resize.target_width or 'auto'}"

    # ------------------------------------------------------------------ persistence
    def _load_saved(self) -> dict:
        key = f"analysis/{self._pipeline_name}"
        raw = self._settings_store.value(key)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            pre = PreprocessingSettings(**data.get("preproc", {})) if data.get("preproc") else None
            edge = EdgeDetectionSettings(**data.get("edge", {})) if data.get("edge") else None
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
        return self._preproc

    def edge_settings(self) -> EdgeDetectionSettings:
        # settings are updated via configuration dialog or internal state
        self._edge.method = self._edge_method.currentText()
        return self._edge

    def pipeline_settings(self) -> Optional[dict]:
        settings = {}
        if self._pipeline_widget and hasattr(self._pipeline_widget, "get_settings"):
            settings = self._pipeline_widget.get_settings() or {}
        
        # Merge all our dynamic tabs
        settings["enabled_stages"] = list(self._enabled_stages)
        settings["physics"] = self._physics_params.model_dump()
        settings["geometry_config"] = self._geometry_config
        settings["overlay_config"] = self._overlay_config
        
        return settings
